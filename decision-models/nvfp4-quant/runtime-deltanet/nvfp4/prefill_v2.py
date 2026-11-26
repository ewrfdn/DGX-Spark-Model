"""P0 + P1 NVFP4 prefill optimization engine for pplx-decider-flash-v1.3-NVFP4.

Combines:
  P0-1: Split-state isolated DeltaNet (eliminates O(Q) full-context recomputation
        and 24x/request CPU-GPU index syncs via precomputed layout + Triton
        token-major isolated causal conv4+SiLU + FLA output_final_state/initial_state).
  P0-2: Native FlashAttention / varlen_attn for the 8 noncausal full-attention
        layers (replaces sm80 mem_efficient fallback caused by 4D bool masks).
  P0-3: FLA 0.6.0 native 16-head GVA (drops Q/K repeat_interleave(2); bitwise equal).
  P0-4: Layer-31 marker-only pruning (computes Q/attn/O/norm/MLP only at readout
        marker rows while keeping full K/V).
  P1:   Layer-scoped shared activation quantisation + inline per-tensor amax from
        producer kernels (fused MLP silu*up+amax, single-CTA residual+RMSNorm+amax,
        single-pass Q/K RMSNorm+partial-RoPE, fused attn*sigmoid(gate)+amax, and
        fused gated-RMSNorm+amax).

All optimizations install cleanly on a live ``NVFP4Decider`` instance and return
a ``restore()`` callable for exact same-process A/B benchmarking.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import torch.nn.functional as F

if __package__:
    from .fp8 import detect_fp8_scaling, fp8_quant_cache_scope
    from .fused_ops import (
        amax_cache_scope,
        fused_add_rmsnorm_amax,
        fused_attn_sigmoid_gate_amax,
        fused_gated_rms_norm_amax,
        fused_isolated_conv4_silu,
        fused_isolated_conv4_silu_split,
        fused_qk_norm_partial_rope,
        fused_rmsnorm_amax,
        fused_silu_mul_amax,
        lookup_amax,
        register_amax,
    )
else:
    from fp8 import detect_fp8_scaling, fp8_quant_cache_scope
    from fused_ops import (
        amax_cache_scope,
        fused_add_rmsnorm_amax,
        fused_attn_sigmoid_gate_amax,
        fused_gated_rms_norm_amax,
        fused_isolated_conv4_silu,
        fused_isolated_conv4_silu_split,
        fused_qk_norm_partial_rope,
        fused_rmsnorm_amax,
        fused_silu_mul_amax,
        lookup_amax,
        register_amax,
    )


@dataclass(frozen=True)
class PrefillV2Config:
    """Toggleable P0/P1 optimization flags for ablation and deployment."""

    # P1: bitwise-identical / producer-fused Triton ops + inline amax + layer shared quant
    shared_quant: bool = True
    inline_amax: bool = True
    fused_mlp_swiglu: bool = True
    fused_residual_rmsnorm: bool = True
    fused_qk_rope_and_gate: bool = True
    # P0: architectural / core-kernel optimizations
    gva_native: bool = True
    split_isolated_deltanet: bool = True
    flash_full_attention: bool = True
    prune_last_layer: bool = True


def prefill_v2_config(mode: str) -> PrefillV2Config | None:
    """Resolve ``DECIDER_PREFILL_V2`` mode string (off|p1|no_prune|full)."""
    key = mode.strip().lower()
    if key in {"", "0", "off", "false", "none"}:
        return None
    if key == "p1":
        return PrefillV2Config(
            gva_native=False,
            split_isolated_deltanet=False,
            flash_full_attention=False,
            prune_last_layer=False,
        )
    if key == "no_prune":
        return PrefillV2Config(prune_last_layer=False)
    if key in {"1", "true", "full", "on"}:
        return PrefillV2Config()
    raise ValueError(f"Unknown DECIDER_PREFILL_V2={mode!r}; expected off, p1, no_prune, or full")


# Context carrier passed from reference_logits() to language_model.forward().
_REQUEST_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "nvfp4_prefill_v2_ctx", default=None
)
_V2_QUANT_CACHE: contextvars.ContextVar[dict[int, tuple[torch.Tensor, tuple[int, ...], Any]] | None] = (
    contextvars.ContextVar("nvfp4_prefill_v2_quant_cache", default=None)
)


# ---------------------------------------------------------------------------
# Precomputed per-request layout (built once on CPU, zero GPU->CPU syncs)
# ---------------------------------------------------------------------------

def build_request_plan(
    chunks: Sequence[Any],
    device: torch.device,
    *,
    isolated: bool,
    conv_taps: int = 4,
) -> dict[str, Any]:
    """Precompute all DeltaNet, Full-Attention, and Layer-31 indices once on CPU."""
    B = len(chunks)
    lengths = [int(c.length) for c in chunks]
    L = max(lengths)
    total = B * L
    has_padding = any(length != L for length in lengths)

    # Marker flat indices (row * L + col) for readout & Layer-31 pruning.
    marker_rows: list[int] = []
    marker_cols: list[int] = []
    marker_flat: list[int] = []
    for r, chunk in enumerate(chunks):
        pad = L - int(chunk.length)
        for m in chunk.markers.tolist():
            col = pad + int(m)
            marker_rows.append(r)
            marker_cols.append(col)
            marker_flat.append(r * L + col)

    # Build token-major conv predecessor table: prev[0]=t-1, prev[1]=t-2, prev[2]=t-3, prev[3]=t (-1 if pad).
    prev = torch.full((conv_taps, total), -1, dtype=torch.int32)

    if not isolated:
        valid_idx_list: list[torch.Tensor] = []
        cu_valid_list: list[int] = [0]
        last_cu_q_list: list[int] = [0]
        last_cu_k_list: list[int] = [0]
        last_kidx_list: list[torch.Tensor] = []
        max_len = max(lengths)
        max_q_per_seq = max(len(c.keys) for c in chunks)

        for r, chunk in enumerate(chunks):
            length = int(chunk.length)
            pad = L - length
            start = r * L + pad
            end = r * L + L
            seq_idx = torch.arange(start, end, dtype=torch.long)
            valid_idx_list.append(seq_idx)
            cu_valid_list.append(cu_valid_list[-1] + length)

            # Conv predecessors within [start, end)
            prev[conv_taps - 1, start:end] = torch.arange(start, end, dtype=torch.int32)
            for shift in range(1, conv_taps):
                if length > shift:
                    prev[shift - 1, start + shift : end] = torch.arange(
                        start, end - shift, dtype=torch.int32
                    )

            # Last layer: each row's markers attend to that row's valid tokens.
            n_q = len(chunk.keys)
            last_cu_q_list.append(last_cu_q_list[-1] + n_q)
            last_cu_k_list.append(last_cu_k_list[-1] + length)
            last_kidx_list.append(seq_idx)

        cu_valid_cpu = torch.tensor(cu_valid_list, dtype=torch.long)
        cu_valid_i32 = cu_valid_cpu.to(device=device, dtype=torch.int32)
        plan: dict[str, Any] = {
            "B": B,
            "L": L,
            "isolated": False,
            "has_padding": has_padding,
            "simple_b1": (B == 1 and not has_padding),
            "conv_prev": prev.to(device=device),
            "valid_idx": torch.cat(valid_idx_list).to(device=device) if has_padding else None,
            "cu_valid_cpu": cu_valid_cpu,
            "cu_valid_long": cu_valid_cpu.to(device=device),
            "cu_valid_i32": cu_valid_i32,
            "max_seqlen": max_len,
            # Layer-31 pruning metadata
            "marker_rows": torch.tensor(marker_rows, dtype=torch.long, device=device),
            "marker_cols": torch.tensor(marker_cols, dtype=torch.long, device=device),
            "marker_flat": torch.tensor(marker_flat, dtype=torch.long, device=device),
            "last_cu_q": torch.tensor(last_cu_q_list, dtype=torch.int32, device=device),
            "last_max_q": max_q_per_seq,
            "last_cu_k": torch.tensor(last_cu_k_list, dtype=torch.int32, device=device),
            "last_max_k": max_len,
            "last_kidx": torch.cat(last_kidx_list).to(device=device) if has_padding else None,
        }
        return plan

    # Isolated mode (one or more chunks have >1 question block)
    state_idx_list: list[torch.Tensor] = []
    block_idx_list: list[torch.Tensor] = []
    state_lens: list[int] = []
    block_lens: list[int] = []
    block_unit: list[int] = []

    fa_qidx_list: list[torch.Tensor] = []
    fa_kidx_list: list[torch.Tensor] = []
    fa_q_lens: list[int] = []
    fa_k_lens: list[int] = []

    last_kidx_list = []
    last_k_lens: list[int] = []

    b1_boundary = 0
    for r, chunk in enumerate(chunks):
        length = int(chunk.length)
        pad = L - length
        start = r * L + pad
        context = int(chunk.segments[0])
        b1_boundary = pad + context
        state_end = start + context
        row_end = start + length

        state_range = torch.arange(start, state_end, dtype=torch.long)
        row_range = torch.arange(start, row_end, dtype=torch.long)

        # DeltaNet state pass
        state_idx_list.append(state_range)
        state_lens.append(context)

        # Conv predecessors for context
        prev[conv_taps - 1, start:state_end] = torch.arange(start, state_end, dtype=torch.int32)
        for shift in range(1, conv_taps):
            if context > shift:
                prev[shift - 1, start + shift : state_end] = torch.arange(
                    start, state_end - shift, dtype=torch.int32
                )

        # Full-attention context query group attends to whole valid row
        fa_qidx_list.append(state_range)
        fa_kidx_list.append(row_range)
        fa_q_lens.append(context)
        fa_k_lens.append(length)

        # Walk question blocks
        begin = state_end
        block_ranges: list[tuple[int, int, torch.Tensor]] = []
        for size in chunk.segments[1:]:
            size = int(size)
            end = begin + size
            blk_range = torch.arange(begin, end, dtype=torch.long)
            block_ranges.append((begin, end, blk_range))

            block_idx_list.append(blk_range)
            block_lens.append(size)
            block_unit.append(r)

            # Conv predecessors for question block: tap reads inside block, or end of context
            prev[conv_taps - 1, begin:end] = torch.arange(begin, end, dtype=torch.int32)
            for shift in range(1, conv_taps):
                if size > shift:
                    prev[shift - 1, begin + shift : end] = torch.arange(
                        begin, end - shift, dtype=torch.int32
                    )
                for offset in range(min(size, shift)):
                    back = context + offset - shift
                    if back >= 0:
                        prev[shift - 1, begin + offset] = start + back

            # Full-attention question block attends to context + itself
            kv_blk = torch.cat((state_range, blk_range))
            fa_qidx_list.append(blk_range)
            fa_kidx_list.append(kv_blk)
            fa_q_lens.append(size)
            fa_k_lens.append(context + size)
            begin = end

        # Map each marker in this chunk to its enclosing question block's KV range
        for m in chunk.markers.tolist():
            m_flat = start + int(m)
            matched = None
            for b_start, b_end, blk_range in block_ranges:
                if b_start <= m_flat < b_end:
                    matched = torch.cat((state_range, blk_range))
                    break
            if matched is None:
                matched = row_range
            last_kidx_list.append(matched)
            last_k_lens.append(int(matched.numel()))

    def _cum(xs: list[int]) -> torch.Tensor:
        return F.pad(torch.tensor(xs, dtype=torch.long).cumsum(0), (1, 0))

    cu_state_cpu = _cum(state_lens)
    cu_block_cpu = _cum(block_lens) if block_lens else torch.zeros(1, dtype=torch.long)
    fa_cu_q = _cum(fa_q_lens).to(device=device, dtype=torch.int32)
    fa_cu_k = _cum(fa_k_lens).to(device=device, dtype=torch.int32)
    last_cu_q = torch.arange(len(marker_flat) + 1, dtype=torch.int32, device=device)
    last_cu_k = _cum(last_k_lens).to(device=device, dtype=torch.int32)

    b1_contiguous = (B == 1 and not has_padding)
    plan = {
        "B": B,
        "L": L,
        "isolated": True,
        "has_padding": has_padding,
        "simple_b1": False,
        "b1_contiguous": b1_contiguous,
        "b1_boundary": b1_boundary,
        "conv_prev": prev.to(device=device),
        # DeltaNet split-state metadata
        "state_idx": None if b1_contiguous else torch.cat(state_idx_list).to(device=device),
        "cu_state_cpu": cu_state_cpu,
        "cu_state_long": cu_state_cpu.to(device=device),
        "block_idx": None if b1_contiguous else (torch.cat(block_idx_list).to(device=device) if block_idx_list else None),
        "cu_block_cpu": cu_block_cpu,
        "cu_block_long": cu_block_cpu.to(device=device),
        "block_unit": torch.tensor(block_unit, dtype=torch.long, device=device),
        "num_blocks": len(block_lens),
        # Full-attention varlen metadata
        "fa_qidx": None if b1_contiguous else torch.cat(fa_qidx_list).to(device=device),
        "fa_kidx": torch.cat(fa_kidx_list).to(device=device),
        "fa_cu_q": fa_cu_q,
        "fa_cu_k": fa_cu_k,
        "fa_max_q": max(fa_q_lens),
        "fa_max_k": max(fa_k_lens),
        # Layer-31 pruning metadata
        "marker_rows": torch.tensor(marker_rows, dtype=torch.long, device=device),
        "marker_cols": torch.tensor(marker_cols, dtype=torch.long, device=device),
        "marker_flat": torch.tensor(marker_flat, dtype=torch.long, device=device),
        "last_cu_q": last_cu_q,
        "last_max_q": 1,
        "last_cu_k": last_cu_k,
        "last_max_k": max(last_k_lens),
        "last_kidx": torch.cat(last_kidx_list).to(device=device),
    }
    return plan


# ---------------------------------------------------------------------------
# Activation quantisation wrapper supporting both shared cache & inline amax
# ---------------------------------------------------------------------------

def _make_quant_patches(config: PrefillV2Config):
    """Build patched activation quantisation hooks compatible with any nvfp4 version."""
    from flashinfer import SfLayout, nvfp4_quantize
    from nvfp4 import backend as nv_backend
    from nvfp4 import linear as nv_linear

    orig_quantize = nv_backend.quantize_activation
    orig_linear_quantize = getattr(nv_linear, "quantize_activation", None)
    orig_cached = getattr(nv_backend, "cached_quantize", None)
    orig_linear_cached = getattr(nv_linear, "cached_quantize", None)

    def quantize_activation_fast(
        x: torch.Tensor,
        *,
        enable_pdl: bool | None = None,
        precomputed_amax: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"quantize_activation expects [M, K], got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"quantize_activation expects bfloat16, got {x.dtype}")
        if not x.is_contiguous():
            x = x.contiguous()

        if precomputed_amax is not None:
            amax = precomputed_amax
        else:
            amax = torch.linalg.vector_norm(x, ord=float("inf")).float().reshape(1)
        torch._assert(torch.isfinite(amax).all(), "NVFP4 activation contains non-finite values")
        is_zero = amax == 0
        safe_amax = torch.where(is_zero, torch.ones_like(amax), amax)
        g_x = torch.where(is_zero, torch.ones_like(amax), nv_backend.AMAX_SCALE / safe_amax)

        if enable_pdl is None:
            enable_pdl = nv_backend._pdl_default()

        x_q, x_sf = nvfp4_quantize(
            x,
            g_x,
            sfLayout=SfLayout.layout_128x4,
            sf_vec_size=nv_backend.BLOCK_SIZE,
            do_shuffle=False,
            backend="cuda",
            per_token_activation=False,
            enable_pdl=enable_pdl,
        )
        return x_q.view(torch.uint8), x_sf.view(torch.uint8), g_x

    def cached_quantize_fast(
        key: torch.Tensor, flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = _V2_QUANT_CACHE.get() if config.shared_quant else None
        if cache is not None:
            entry = cache.get(id(key))
            if entry is not None and entry[0] is key and entry[1] == flat.shape:
                return entry[2]

        amax = lookup_amax(key, (flat.shape[0], flat.shape[1])) if config.inline_amax else None
        # Call through nv_backend.quantize_activation so profilers wrapping it still see every call.
        if amax is not None:
            value = nv_backend.quantize_activation(flat, precomputed_amax=amax)
        else:
            value = nv_backend.quantize_activation(flat)

        if cache is not None:
            cache[id(key)] = (key, flat.shape, value)
        return value

    def linear_quantize_wrapper(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fallback for older nvfp4/linear.py calling quantize_activation(x2) directly.
        return cached_quantize_fast(flat, flat)

    def install() -> Callable[[], None]:
        nv_backend.quantize_activation = quantize_activation_fast
        if orig_cached is not None:
            nv_backend.cached_quantize = cached_quantize_fast
        if orig_linear_cached is not None:
            nv_linear.cached_quantize = cached_quantize_fast
        if orig_linear_quantize is not None:
            nv_linear.quantize_activation = linear_quantize_wrapper

        def restore() -> None:
            nv_backend.quantize_activation = orig_quantize
            if orig_cached is not None:
                nv_backend.cached_quantize = orig_cached
            if orig_linear_cached is not None:
                nv_linear.cached_quantize = orig_linear_cached
            if orig_linear_quantize is not None:
                nv_linear.quantize_activation = orig_linear_quantize

        return restore

    return install


# ---------------------------------------------------------------------------
# Optimized mixers & MLP
# ---------------------------------------------------------------------------

def _run_mlp(mlp: torch.nn.Module, x: torch.Tensor, config: PrefillV2Config, *, compute_amax: bool = True) -> torch.Tensor:
    if not hasattr(mlp, "gate_proj"):
        return mlp(x)
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    if config.fused_mlp_swiglu:
        mid, _ = fused_silu_mul_amax(gate, up, compute_amax=compute_amax)
    else:
        mid = mlp.act_fn(gate) * up
    return mlp.down_proj(mid)


def _run_deltanet(
    mod: torch.nn.Module,
    hidden_states: torch.Tensor,
    plan: dict[str, Any],
    linear_mask: torch.Tensor | None,
    config: PrefillV2Config,
    chunk_rule_fn: Callable,
    orig_deltanet_forward: Callable,
    scope_fn: Callable[[str], Any] = lambda _: contextlib.nullcontext(),
    *,
    compute_amax: bool = True,
) -> torch.Tensor:
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    if not config.split_isolated_deltanet and plan["isolated"]:
        # Fall back to original (restarted) path if split_isolated is explicitly disabled.
        return orig_deltanet_forward(hidden_states, attention_mask=linear_mask)

    with scope_fn("delta_layout_and_gates"):
        if linear_mask is not None:
            hidden_states = apply_mask_to_padding_states(hidden_states, linear_mask)

    B, T, _ = hidden_states.shape

    with scope_fn("delta_projection_other"):
        mixed_qkv = mod.in_proj_qkv(hidden_states)  # [B, T, 8192] contiguous
        z = mod.in_proj_z(hidden_states).reshape(B, T, -1, mod.head_v_dim)
        b = mod.in_proj_b(hidden_states)
        a = mod.in_proj_a(hidden_states)

    # Token-major depthwise causal conv (k=4) + SiLU, writing contiguous (q, k, v) directly.
    with scope_fn("causal_conv"):
        if mod.conv1d.bias is None and mod.conv_kernel_size == 4 and mod.activation == "silu":
            q_flat, k_flat, v_flat = fused_isolated_conv4_silu_split(
                mixed_qkv,
                mod.conv1d.weight.squeeze(1),
                plan["conv_prev"],
                mod.key_dim,
                mod.value_dim,
            )
            query = q_flat.view(B, T, -1, mod.head_k_dim)
            key = k_flat.view(B, T, -1, mod.head_k_dim)
            value = v_flat.view(B, T, -1, mod.head_v_dim)
        else:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

            mixed_qkv = qwen.causal_conv1d_fn(
                mixed_qkv.transpose(1, 2),
                mod.conv1d.weight.squeeze(1),
                mod.conv1d.bias,
                activation=mod.activation,
            ).transpose(1, 2)
            q_flat, k_flat, v_flat = torch.split(mixed_qkv, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
            query = q_flat.reshape(B, T, -1, mod.head_k_dim)
            key = k_flat.reshape(B, T, -1, mod.head_k_dim)
            value = v_flat.reshape(B, T, -1, mod.head_v_dim)

    with scope_fn("delta_layout_and_gates"):
        beta = b.sigmoid()
        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)

        if not config.gva_native and (mod.num_v_heads // mod.num_k_heads) > 1:
            rep = mod.num_v_heads // mod.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

    with scope_fn("gdn_core"):
        if not plan["isolated"]:
            if not plan["has_padding"] or not config.split_isolated_deltanet:
                core, _ = chunk_rule_fn(
                    query, key, value, g=g, beta=beta,
                    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
                )
            else:
                vi = plan["valid_idx"]
                qf = query.reshape(B * T, -1, mod.head_k_dim)[vi].unsqueeze(0)
                kf = key.reshape(B * T, -1, mod.head_k_dim)[vi].unsqueeze(0)
                vf = value.reshape(B * T, -1, mod.head_v_dim)[vi].unsqueeze(0)
                gf = g.reshape(B * T, -1)[vi].unsqueeze(0)
                bf = beta.reshape(B * T, -1)[vi].unsqueeze(0)
                packed_out, _ = chunk_rule_fn(
                    qf, kf, vf, g=gf, beta=bf,
                    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=plan["cu_valid_long"], cu_seqlens_cpu=plan["cu_valid_cpu"],
                )
                core = torch.zeros((B * T, mod.num_v_heads, mod.head_v_dim), dtype=value.dtype, device=value.device)
                core[vi] = packed_out[0]
                core = core.view(B, T, mod.num_v_heads, mod.head_v_dim)
        else:
            if plan["b1_contiguous"]:
                bnd = plan["b1_boundary"]
                core = torch.empty_like(value)
                o_ctx, final_state = chunk_rule_fn(
                    query[:, :bnd], key[:, :bnd], value[:, :bnd],
                    g=g[:, :bnd], beta=beta[:, :bnd],
                    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
                )
                core[:, :bnd] = o_ctx
                if bnd < T:
                    init = final_state[plan["block_unit"]]
                    o_blk, _ = chunk_rule_fn(
                        query[:, bnd:], key[:, bnd:], value[:, bnd:],
                        g=g[:, bnd:], beta=beta[:, bnd:],
                        initial_state=init, output_final_state=False, use_qk_l2norm_in_kernel=True,
                        cu_seqlens=plan["cu_block_long"], cu_seqlens_cpu=plan["cu_block_cpu"],
                    )
                    core[:, bnd:] = o_blk
            else:
                si = plan["state_idx"]
                bi = plan["block_idx"]
                qf = query.reshape(B * T, -1, mod.head_k_dim)
                kf = key.reshape(B * T, -1, mod.head_k_dim)
                vf = value.reshape(B * T, -1, mod.head_v_dim)
                gf = g.reshape(B * T, -1)
                bf = beta.reshape(B * T, -1)

                core_flat = torch.zeros((B * T, mod.num_v_heads, mod.head_v_dim), dtype=value.dtype, device=value.device)
                o_ctx, final_state = chunk_rule_fn(
                    qf[si].unsqueeze(0), kf[si].unsqueeze(0), vf[si].unsqueeze(0),
                    g=gf[si].unsqueeze(0), beta=bf[si].unsqueeze(0),
                    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=plan["cu_state_long"], cu_seqlens_cpu=plan["cu_state_cpu"],
                )
                core_flat[si] = o_ctx[0]
                if bi is not None and bi.numel() > 0:
                    init = final_state[plan["block_unit"]]
                    o_blk, _ = chunk_rule_fn(
                        qf[bi].unsqueeze(0), kf[bi].unsqueeze(0), vf[bi].unsqueeze(0),
                        g=gf[bi].unsqueeze(0), beta=bf[bi].unsqueeze(0),
                        initial_state=init, output_final_state=False, use_qk_l2norm_in_kernel=True,
                        cu_seqlens=plan["cu_block_long"], cu_seqlens_cpu=plan["cu_block_cpu"],
                    )
                    core_flat[bi] = o_blk[0]
                core = core_flat.view(B, T, mod.num_v_heads, mod.head_v_dim)

    with scope_fn("gated_rmsnorm"):
        normed, _ = fused_gated_rms_norm_amax(
            core, z, mod.norm.weight, mod.norm.variance_epsilon,
            out_shape=(B, T, mod.value_dim), compute_amax=compute_amax,
        )
    with scope_fn("delta_projection_other"):
        return mod.out_proj(normed)


def _run_full_attention(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: dict[str, Any],
    full_mask: torch.Tensor | None,
    config: PrefillV2Config,
    varlen_attn_fn: Callable,
    flash_attn_fn: Callable | None,
    scope_fn: Callable[[str], Any] = lambda _: contextlib.nullcontext(),
    *,
    compute_amax: bool = True,
) -> torch.Tensor:
    B, T, _ = hidden_states.shape
    hd = attn.head_dim

    with scope_fn("full_attention_other"):
        qg = attn.q_proj(hidden_states).view(B, T, -1, 2, hd)
        q_raw = qg[:, :, :, 0, :]
        gate = qg[:, :, :, 1, :]

        k_raw = attn.k_proj(hidden_states).view(B, T, -1, hd)
        v = attn.v_proj(hidden_states).view(B, T, -1, hd)

        if config.fused_qk_rope_and_gate:
            q = fused_qk_norm_partial_rope(q_raw, attn.q_norm.weight, cos, sin, attn.q_norm.eps)
            k = fused_qk_norm_partial_rope(k_raw, attn.k_norm.weight, cos, sin, attn.k_norm.eps)
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

            q_h = attn.q_norm(q_raw).transpose(1, 2)
            k_h = attn.k_norm(k_raw).transpose(1, 2)
            q_h, k_h = apply_rotary_pos_emb(q_h, k_h, cos, sin)
            q = q_h.transpose(1, 2).contiguous()
            k = k_h.transpose(1, 2).contiguous()

    with scope_fn("sdpa"):
        if config.flash_full_attention:
            gqa = k.shape[2] != q.shape[2]
            if plan["simple_b1"]:
                if flash_attn_fn is not None:
                    attn_out = flash_attn_fn(q, k, v, softmax_scale=attn.scaling, causal=False)
                else:
                    cu = plan["cu_valid_i32"]
                    attn_out = varlen_attn_fn(
                        q[0], k[0], v[0], cu, cu, T, T,
                        scale=attn.scaling, window_size=(-1, -1), enable_gqa=gqa,
                    ).unsqueeze(0)
            elif not plan["isolated"]:
                if not plan["has_padding"] and flash_attn_fn is not None:
                    attn_out = flash_attn_fn(q, k, v, softmax_scale=attn.scaling, causal=False)
                else:
                    vi = plan["valid_idx"]
                    qf = q.reshape(B * T, q.shape[2], hd)
                    kf = k.reshape(B * T, k.shape[2], hd)
                    vf = v.reshape(B * T, v.shape[2], hd)
                    if vi is not None:
                        qf, kf, vf = qf[vi], kf[vi], vf[vi]
                    cu = plan["cu_valid_i32"]
                    packed = varlen_attn_fn(
                        qf, kf, vf, cu, cu, plan["max_seqlen"], plan["max_seqlen"],
                        scale=attn.scaling, window_size=(-1, -1), enable_gqa=gqa,
                    )
                    if vi is not None:
                        out_flat = torch.zeros((B * T, q.shape[2], hd), dtype=q.dtype, device=q.device)
                        out_flat[vi] = packed
                        attn_out = out_flat.view(B, T, q.shape[2], hd)
                    else:
                        attn_out = packed.view(B, T, q.shape[2], hd)
            else:
                kidx = plan["fa_kidx"]
                qf = q.reshape(B * T, q.shape[2], hd)
                kf = k.reshape(B * T, k.shape[2], hd)
                vf = v.reshape(B * T, v.shape[2], hd)
                qidx = plan["fa_qidx"]
                q_in = qf if qidx is None else qf[qidx]
                packed = varlen_attn_fn(
                    q_in, kf[kidx], vf[kidx],
                    plan["fa_cu_q"], plan["fa_cu_k"], plan["fa_max_q"], plan["fa_max_k"],
                    scale=attn.scaling, window_size=(-1, -1), enable_gqa=gqa,
                )
                if qidx is None:
                    attn_out = packed.view(B, T, q.shape[2], hd)
                else:
                    out_flat = torch.zeros((B * T, q.shape[2], hd), dtype=q.dtype, device=q.device)
                    out_flat[qidx] = packed
                    attn_out = out_flat.view(B, T, q.shape[2], hd)
        else:
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            if hasattr(attn, "num_key_value_groups") and attn.num_key_value_groups > 1:
                k_t = k_t.repeat_interleave(attn.num_key_value_groups, dim=1)
                v_t = v_t.repeat_interleave(attn.num_key_value_groups, dim=1)
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=full_mask, dropout_p=0.0, scale=attn.scaling, is_causal=False,
            ).transpose(1, 2).contiguous()

    with scope_fn("full_attention_other"):
        if config.fused_qk_rope_and_gate:
            gated, _ = fused_attn_sigmoid_gate_amax(attn_out, gate, compute_amax=compute_amax)
        else:
            gated = attn_out.reshape(B, T, -1).contiguous() * torch.sigmoid(gate.reshape(B, T, -1))
        return attn.o_proj(gated)


def _run_pruned_last_layer(
    layer: torch.nn.Module,
    final_norm: torch.nn.Module,
    residual: torch.Tensor,
    normed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: dict[str, Any],
    config: PrefillV2Config,
    varlen_attn_fn: Callable,
    scope_fn: Callable[[str], Any] = lambda _: contextlib.nullcontext(),
) -> torch.Tensor:
    """Layer 31 (full_attention) + final norm evaluated only at readout marker rows."""
    attn = layer.self_attn
    B, T, D = residual.shape
    hd = attn.head_dim

    with scope_fn("full_attention_other"):
        k_raw = attn.k_proj(normed).view(B, T, -1, hd)
        v = attn.v_proj(normed).view(B, T, -1, hd)
        if config.fused_qk_rope_and_gate:
            k = fused_qk_norm_partial_rope(k_raw, attn.k_norm.weight, cos, sin, attn.k_norm.eps)
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

            k_h = attn.k_norm(k_raw).transpose(1, 2)
            _, k_h = apply_rotary_pos_emb(k_h, k_h, cos, sin)
            k = k_h.transpose(1, 2).contiguous()

        m_rows = plan["marker_rows"]
        m_cols = plan["marker_cols"]
        x_sel = normed[m_rows, m_cols].unsqueeze(0).contiguous()
        h_sel = residual[m_rows, m_cols].unsqueeze(0).contiguous()
        cos_sel = cos[m_rows, m_cols].unsqueeze(0).contiguous()
        sin_sel = sin[m_rows, m_cols].unsqueeze(0).contiguous()
        M_sel = x_sel.shape[1]

        qg = attn.q_proj(x_sel).view(1, M_sel, -1, 2, hd)
        q_raw = qg[:, :, :, 0, :]
        gate = qg[:, :, :, 1, :]
        q_sel = fused_qk_norm_partial_rope(q_raw, attn.q_norm.weight, cos_sel, sin_sel, attn.q_norm.eps)

    with scope_fn("sdpa"):
        kf = k.reshape(B * T, k.shape[2], hd)
        vf = v.reshape(B * T, v.shape[2], hd)
        kidx = plan["last_kidx"]
        if kidx is not None:
            kf = kf[kidx]
            vf = vf[kidx]
        gqa = kf.shape[1] != q_sel.shape[2]
        attn_out = varlen_attn_fn(
            q_sel[0], kf, vf,
            plan["last_cu_q"], plan["last_cu_k"], plan["last_max_q"], plan["last_max_k"],
            scale=attn.scaling, window_size=(-1, -1), enable_gqa=gqa,
        ).unsqueeze(0)

    with scope_fn("full_attention_other"):
        gated = attn_out.reshape(1, M_sel, -1).contiguous() * torch.sigmoid(gate.reshape(1, M_sel, -1))
        o_out = attn.o_proj(gated)

    with scope_fn("plain_rmsnorm"):
        h_sel = h_sel + o_out
        mlp_in = layer.post_attention_layernorm(h_sel)
    with scope_fn("mlp_other"):
        if hasattr(layer.mlp, "gate_proj"):
            mlp_out = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(mlp_in)) * layer.mlp.up_proj(mlp_in))
        else:
            mlp_out = layer.mlp(mlp_in)
    with scope_fn("plain_rmsnorm"):
        h_sel = h_sel + mlp_out
        final_sel = final_norm(h_sel)[0]

    out = torch.empty_like(residual)
    out[m_rows, m_cols] = final_sel
    return out


# ---------------------------------------------------------------------------
# Installer on NVFP4 / FP8 / BF16 Decider (cleanly reversible for A/B)
# ---------------------------------------------------------------------------

def install_prefill_v2(
    decider: Any,
    config: PrefillV2Config | None = None,
    *,
    profile_scopes: bool = False,
) -> Callable[[], None]:
    """Install P0+P1 prefill optimizations on ``decider`` (NVFP4, FP8, or BF16) and return ``restore()``."""
    cfg = config or PrefillV2Config()

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from torch.nn.attention.varlen import varlen_attn
    from torch.profiler import record_function
    from transformers.modeling_outputs import BaseModelOutputWithPast

    backbone = decider.backbone
    text_model = backbone.language_model
    layers = list(text_model.layers)

    is_nvfp4 = hasattr(decider, "nvfp4_checkpoint")
    fp8_scaling = None if is_nvfp4 else detect_fp8_scaling(backbone)
    effective_nvfp4_quant = cfg.shared_quant and is_nvfp4
    effective_fp8_quant = cfg.shared_quant and (fp8_scaling is not None)
    effective_inline_amax = cfg.inline_amax and (is_nvfp4 or fp8_scaling == "tensor")

    @contextlib.contextmanager
    def _layer_quant_scope():
        if effective_nvfp4_quant:
            token = _V2_QUANT_CACHE.set({})
            try:
                yield
            finally:
                _V2_QUANT_CACHE.reset(token)
        elif effective_fp8_quant:
            with fp8_quant_cache_scope(enabled=True):
                yield
        else:
            yield

    def _scope(name: str):
        return record_function("prefill::" + name) if profile_scopes else contextlib.nullcontext()

    try:
        import flash_attn

        flash_attn_fn = flash_attn.flash_attn_func
    except Exception:
        flash_attn_fn = None

    # 1. Patch NVFP4 activation quantisation cache + inline amax support (NVFP4 only).
    restore_quant = _make_quant_patches(cfg)() if is_nvfp4 else (lambda: None)

    # 2. Wrap decider.reference_logits to supply chunk layout & marker positions on CPU.
    orig_ref_logits = vars(decider).get("reference_logits")
    bound_ref_logits = decider.reference_logits

    @wraps(bound_ref_logits)
    def wrapped_reference_logits(chunks):
        isolated = any(len(c.keys) > 1 for c in chunks)
        plan = build_request_plan(chunks, decider.device, isolated=isolated, conv_taps=4)
        token = _REQUEST_CTX.set(plan)
        try:
            return bound_ref_logits(chunks)
        finally:
            _REQUEST_CTX.reset(token)

    decider.reference_logits = wrapped_reference_logits

    # 3. Wrap text_model.forward to execute the optimized 32-layer loop.
    orig_text_forward = vars(text_model).get("forward")
    bound_text_forward = text_model.forward

    @wraps(bound_text_forward)
    def fast_text_forward(
        input_ids: torch.LongTensor | None = None,
        attention_mask: Any = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ):
        plan = _REQUEST_CTX.get()
        if (
            plan is None
            or use_cache
            or past_key_values is not None
            or torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or kwargs.get("output_attentions")
            or kwargs.get("output_hidden_states")
        ):
            return bound_text_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            rope_pos = position_ids[1:]
        else:
            rope_pos = position_ids

        # Masks prepared by Decider's pre-hooks (enable_noncausal_full_attention + IsolatedQuestions)
        if isinstance(attention_mask, dict):
            full_mask = attention_mask.get("full_attention")
            linear_mask = attention_mask.get("linear_attention")
        else:
            full_mask = None
            linear_mask = None
        if plan["isolated"] and decider.isolation.layout is not None:
            full_mask = decider.isolation.layout.mask

        hidden_states = inputs_embeds
        cos, sin = text_model.rotary_emb(hidden_states, rope_pos)

        num_layers = len(layers)
        prune_last = (
            cfg.prune_last_layer
            and cfg.flash_full_attention
            and num_layers >= 1
            and layers[-1].block_type == "full_attention"
        )
        main_count = num_layers - 1 if prune_last else num_layers

        # Initial input_layernorm for layer 0
        with _scope("layer:0"), _scope("delta_decoder" if layers[0].block_type == "linear_attention" else "full_decoder"), _scope("plain_rmsnorm"):
            if cfg.fused_residual_rmsnorm:
                normed, next_amax = fused_rmsnorm_amax(
                    hidden_states, layers[0].input_layernorm.weight, layers[0].input_layernorm.eps,
                    compute_amax=effective_inline_amax,
                )
            else:
                normed = layers[0].input_layernorm(hidden_states)
                next_amax = None

        for i in range(main_count):
            layer = layers[i]
            layer_kind_scope = "delta_decoder" if layer.block_type == "linear_attention" else "full_decoder"
            # Scope BOTH quant cache and amax cache to a single layer so intermediates are freed immediately.
            with _scope(f"layer:{i}"), _scope(layer_kind_scope), _layer_quant_scope(), amax_cache_scope(enabled=effective_inline_amax):
                if next_amax is not None:
                    register_amax(normed, next_amax)
                if layer.block_type == "linear_attention":
                    mixer_out = _run_deltanet(
                        layer.linear_attn,
                        normed,
                        plan,
                        linear_mask,
                        cfg,
                        chunk_gated_delta_rule,
                        layer.linear_attn.forward,
                        _scope,
                        compute_amax=effective_inline_amax,
                    )
                else:
                    mixer_out = _run_full_attention(
                        layer.self_attn,
                        normed,
                        cos,
                        sin,
                        plan,
                        full_mask,
                        cfg,
                        varlen_attn,
                        flash_attn_fn,
                        _scope,
                        compute_amax=effective_inline_amax,
                    )

                # Residual add #1 + post_attention_layernorm (+ inline amax)
                with _scope("plain_rmsnorm"):
                    if cfg.fused_residual_rmsnorm:
                        hidden_states, post_normed, _ = fused_add_rmsnorm_amax(
                            hidden_states,
                            mixer_out,
                            layer.post_attention_layernorm.weight,
                            layer.post_attention_layernorm.eps,
                            compute_amax=effective_inline_amax,
                        )
                    else:
                        hidden_states = hidden_states + mixer_out
                        post_normed = layer.post_attention_layernorm(hidden_states)

                # MLP (with fused silu*up + inline amax)
                with _scope("mlp_other"):
                    mlp_out = _run_mlp(layer.mlp, post_normed, cfg, compute_amax=effective_inline_amax)

                # Residual add #2 + next layer's input_layernorm (or final norm)
                with _scope("plain_rmsnorm"):
                    if i + 1 < num_layers:
                        next_norm = layers[i + 1].input_layernorm
                        if cfg.fused_residual_rmsnorm:
                            hidden_states, normed, next_amax = fused_add_rmsnorm_amax(
                                hidden_states, mlp_out, next_norm.weight, next_norm.eps,
                                compute_amax=effective_inline_amax,
                            )
                        else:
                            hidden_states = hidden_states + mlp_out
                            normed = next_norm(hidden_states)
                            next_amax = None
                    else:
                        if cfg.fused_residual_rmsnorm:
                            hidden_states, final_hidden, _ = fused_add_rmsnorm_amax(
                                hidden_states, mlp_out, text_model.norm.weight, text_model.norm.eps,
                                compute_amax=False,
                            )
                        else:
                            hidden_states = hidden_states + mlp_out
                            final_hidden = text_model.norm(hidden_states)

        if prune_last:
            last_idx = num_layers - 1
            with _scope(f"layer:{last_idx}"), _scope("full_decoder"), _layer_quant_scope(), amax_cache_scope(enabled=effective_inline_amax):
                if next_amax is not None:
                    register_amax(normed, next_amax)
                final_hidden = _run_pruned_last_layer(
                    layers[-1],
                    text_model.norm,
                    hidden_states,
                    normed,
                    cos,
                    sin,
                    plan,
                    cfg,
                    varlen_attn,
                    _scope,
                )

        return BaseModelOutputWithPast(last_hidden_state=final_hidden, past_key_values=None)

    text_model.forward = fast_text_forward

    def restore() -> None:
        if orig_text_forward is None:
            if "forward" in vars(text_model):
                del text_model.forward
        else:
            text_model.forward = orig_text_forward
        if orig_ref_logits is None:
            if "reference_logits" in vars(decider):
                del decider.reference_logits
        else:
            decider.reference_logits = orig_ref_logits
        restore_quant()

    return restore
