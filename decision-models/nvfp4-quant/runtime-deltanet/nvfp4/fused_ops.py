"""Fused Triton prefill kernels with inline per-tensor amax for NVFP4 W4A4.

All kernels preserve eager ATen BF16/FP32 rounding and reduction semantics so
outputs remain bit-identical (or within 1 ULP where noted) while eliminating
redundant HBM sweeps and all standalone ``absmax`` reductions before
``nvfp4_quantize``.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# ---------------------------------------------------------------------------
# Scope-local precomputed amax cache consumed by NVFP4 activation quantisation
# ---------------------------------------------------------------------------

_AMAX_CACHE: contextvars.ContextVar[dict[int, tuple[torch.Tensor, tuple[int, ...], torch.Tensor]] | None] = (
    contextvars.ContextVar("nvfp4_amax_cache", default=None)
)


@contextlib.contextmanager
def amax_cache_scope(*, enabled: bool = True):
    token = _AMAX_CACHE.set({} if enabled else None)
    try:
        yield
    finally:
        _AMAX_CACHE.reset(token)


def register_amax(tensor: torch.Tensor, amax: torch.Tensor) -> None:
    """Attach a precomputed FP32 scalar ``amax`` (shape ``[1]``) to ``tensor``."""
    cache = _AMAX_CACHE.get()
    if cache is None:
        return
    cache[id(tensor)] = (tensor, tuple(tensor.shape), amax)


def lookup_amax(key: torch.Tensor, flat_shape: tuple[int, int]) -> torch.Tensor | None:
    """Return cached FP32 ``amax`` if ``key`` matches identity and element count."""
    cache = _AMAX_CACHE.get()
    if cache is None:
        return None
    entry = cache.get(id(key))
    if entry is None:
        return None
    cached_tensor, cached_shape, amax = entry
    if cached_tensor is not key:
        return None
    if math.prod(cached_shape) != flat_shape[0] * flat_shape[1]:
        return None
    return amax


# ---------------------------------------------------------------------------
# Helper: NaN-preserving unsigned-magnitude int32 bit pattern for atomic_max
# ---------------------------------------------------------------------------

@triton.jit
def _abs_f32_to_sortable_i32(x_f32):
    # Mask off the sign bit. For non-negative IEEE-754 float32 values:
    #   +0.0 -> 0x00000000
    #   normal/subnormal -> strictly monotonic with magnitude
    #   +inf -> 0x7F800000
    #   NaN  -> > 0x7F800000 (always wins integer max, never dropped)
    return x_f32.to(tl.int32, bitcast=True) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# 1. Fused MLP SwiGLU: out = (silu(gate).bf16 * up).bf16 + inline amax
# ---------------------------------------------------------------------------

@triton.jit
def _silu_mul_amax_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    amax_i32_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Match ATen BF16 silu (rounded to BF16) followed by BF16 mul.
    s_bf16 = tl.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16)
    y_bf16 = (s_bf16.to(tl.float32) * u).to(tl.bfloat16)
    tl.store(out_ptr + offs, y_bf16, mask=mask)

    y_f32 = tl.where(mask, y_bf16.to(tl.float32), 0.0)
    block_max = tl.max(_abs_f32_to_sortable_i32(y_f32), axis=0)
    tl.atomic_max(amax_i32_ptr, block_max)


def fused_silu_mul_amax(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact ``F.silu(gate) * up`` and its BF16 per-tensor FP32 ``amax``."""
    if not (
        gate.is_cuda
        and gate.dtype == up.dtype == torch.bfloat16
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
    ):
        out = torch.nn.functional.silu(gate) * up
        amax = torch.linalg.vector_norm(out, ord=float("inf")).float().reshape(1)
        return out, amax

    out = torch.empty_like(gate)
    n = gate.numel()
    if n == 0:
        return out, torch.zeros(1, dtype=torch.float32, device=gate.device)

    amax_i32 = torch.zeros(1, dtype=torch.int32, device=gate.device)
    block = 2048
    with torch.cuda.device(gate.device):
        _silu_mul_amax_kernel[(triton.cdiv(n, block),)](
            gate, up, out, amax_i32, n, BLOCK=block, num_warps=8, enable_fp_fusion=False,
        )
    amax = amax_i32.view(torch.float32)
    register_amax(out, amax)
    return out, amax


# ---------------------------------------------------------------------------
# 2. Single-CTA Residual-Add + exact-order Plain RMSNorm + inline amax
# ---------------------------------------------------------------------------

@triton.jit
def _rmsnorm_2560_amax_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    amax_i32_ptr,
    rows,
    x_row_stride,
    eps: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.arange(0, 128)
    valid = row[:, None] < rows
    row_i64 = row[:, None].to(tl.int64)
    x_base = row_i64 * x_row_stride + col[None, :]
    out_base = row_i64 * width + col[None, :]

    # Pass 1: exact ATen float4 vectorized reduction order across 20 waves of 128.
    first = tl.full((block_rows, 32), 0.0, tl.float32)
    second = tl.full((block_rows, 32), 0.0, tl.float32)
    third = tl.full((block_rows, 32), 0.0, tl.float32)
    fourth = tl.full((block_rows, 32), 0.0, tl.float32)
    for wave in tl.range(width // 128, loop_unroll_factor=2):
        vals = tl.load(x_ptr + x_base + wave * 128, valid, other=0.0).to(tl.float32)
        even, odd = tl.split(tl.reshape(vals * vals, (block_rows, 32, 2, 2)))
        a, c = tl.split(even)
        b, d = tl.split(odd)
        first += a
        second += b
        third += c
        fourth += d
    total = tl.sum(((first + second) + third) + fourth, axis=1)
    inv_rms = tl.rsqrt(total * (1.0 / width) + eps)

    # Pass 2: x row (5 KB) is hot in L1/L2 cache; apply weight and track amax.
    max_bits = tl.full((), 0, tl.int32)
    for wave in tl.range(width // 128, loop_unroll_factor=2):
        off = wave * 128
        vals = tl.load(x_ptr + x_base + off, valid, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col + off).to(tl.float32)
        y_bf16 = ((vals * inv_rms[:, None]) * (1.0 + w[None, :])).to(tl.bfloat16)
        tl.store(out_ptr + out_base + off, y_bf16, valid)
        if COMPUTE_AMAX:
            y_f32 = tl.where(valid, y_bf16.to(tl.float32), 0.0)
            wave_max = tl.max(_abs_f32_to_sortable_i32(y_f32))
            max_bits = tl.maximum(max_bits, wave_max)

    if COMPUTE_AMAX:
        tl.atomic_max(amax_i32_ptr, max_bits)


@triton.jit
def _add_rmsnorm_2560_amax_kernel(
    res_ptr,
    delta_ptr,
    weight_ptr,
    sum_out_ptr,
    norm_out_ptr,
    amax_i32_ptr,
    rows,
    eps: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.arange(0, 128)
    valid = row[:, None] < rows
    base = row[:, None].to(tl.int64) * width + col[None, :]

    # Pass 1: h = (residual + delta).bf16 -> store h (stays hot in L1/L2) + exact ATen variance sum.
    first = tl.full((block_rows, 32), 0.0, tl.float32)
    second = tl.full((block_rows, 32), 0.0, tl.float32)
    third = tl.full((block_rows, 32), 0.0, tl.float32)
    fourth = tl.full((block_rows, 32), 0.0, tl.float32)
    for wave in tl.range(width // 128, loop_unroll_factor=2):
        off = wave * 128
        r = tl.load(res_ptr + base + off, valid, other=0.0).to(tl.float32)
        d = tl.load(delta_ptr + base + off, valid, other=0.0).to(tl.float32)
        h_bf16 = (r + d).to(tl.bfloat16)
        tl.store(sum_out_ptr + base + off, h_bf16, valid)
        h_f32 = h_bf16.to(tl.float32)
        even, odd = tl.split(tl.reshape(h_f32 * h_f32, (block_rows, 32, 2, 2)))
        a, c = tl.split(even)
        b, d_ = tl.split(odd)
        first += a
        second += b
        third += c
        fourth += d_
    total = tl.sum(((first + second) + third) + fourth, axis=1)
    inv_rms = tl.rsqrt(total * (1.0 / width) + eps)

    # Pass 2: read h back from L1/L2, normalize, store, and track amax.
    max_bits = tl.full((), 0, tl.int32)
    for wave in tl.range(width // 128, loop_unroll_factor=2):
        off = wave * 128
        h_f32 = tl.load(sum_out_ptr + base + off, valid, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col + off).to(tl.float32)
        y_bf16 = ((h_f32 * inv_rms[:, None]) * (1.0 + w[None, :])).to(tl.bfloat16)
        tl.store(norm_out_ptr + base + off, y_bf16, valid)
        if COMPUTE_AMAX:
            y_f32 = tl.where(valid, y_bf16.to(tl.float32), 0.0)
            wave_max = tl.max(_abs_f32_to_sortable_i32(y_f32))
            max_bits = tl.maximum(max_bits, wave_max)

    if COMPUTE_AMAX:
        tl.atomic_max(amax_i32_ptr, max_bits)


def fused_rmsnorm_amax(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    compute_amax: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Single-CTA exact-order plain RMSNorm for width=2560, with optional inline amax."""
    width = x.shape[-1]
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or width != 2560
        or weight.shape != (2560,)
        or not weight.is_contiguous()
        or x.stride(-1) != 1
    ):
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)
        amax = torch.linalg.vector_norm(out, ord=float("inf")).float().reshape(1) if compute_amax else None
        if amax is not None:
            register_amax(out, amax)
        return out, amax

    flat = x.reshape(-1, width)
    rows = flat.shape[0]
    if rows < 16:
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)
        amax = torch.linalg.vector_norm(out, ord=float("inf")).float().reshape(1) if compute_amax else None
        if amax is not None:
            register_amax(out, amax)
        return out, amax

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    amax_i32 = torch.zeros(1, dtype=torch.int32, device=x.device) if compute_amax else torch.empty(1, dtype=torch.int32, device=x.device)
    block_rows = 2
    with torch.cuda.device(x.device):
        _rmsnorm_2560_amax_kernel[(triton.cdiv(rows, block_rows),)](
            flat,
            weight,
            out,
            amax_i32,
            rows,
            flat.stride(0),
            eps=eps,
            width=2560,
            block_rows=block_rows,
            COMPUTE_AMAX=compute_amax,
            num_warps=4,
            enable_fp_fusion=False,
        )
    amax = amax_i32.view(torch.float32) if compute_amax else None
    if amax is not None:
        register_amax(out, amax)
    return out, amax


def fused_add_rmsnorm_amax(
    residual: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    compute_amax: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute ``h = residual + delta`` and ``normed = RMSNorm(h)`` (+ ``amax``) in one CTA."""
    width = residual.shape[-1]
    if (
        not residual.is_cuda
        or residual.dtype != delta.dtype == weight.dtype == torch.bfloat16
        or residual.shape != delta.shape
        or width != 2560
        or weight.shape != (2560,)
        or not (residual.is_contiguous() and delta.is_contiguous() and weight.is_contiguous())
    ):
        h = residual + delta
        normed, amax = fused_rmsnorm_amax(h, weight, eps, compute_amax=compute_amax)
        return h, normed, amax

    rows = residual.numel() // width
    if rows < 16:
        h = residual + delta
        normed, amax = fused_rmsnorm_amax(h, weight, eps, compute_amax=compute_amax)
        return h, normed, amax

    h_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)
    amax_i32 = torch.zeros(1, dtype=torch.int32, device=residual.device) if compute_amax else torch.empty(1, dtype=torch.int32, device=residual.device)
    block_rows = 2
    with torch.cuda.device(residual.device):
        _add_rmsnorm_2560_amax_kernel[(triton.cdiv(rows, block_rows),)](
            residual,
            delta,
            weight,
            h_out,
            norm_out,
            amax_i32,
            rows,
            eps=eps,
            width=2560,
            block_rows=block_rows,
            COMPUTE_AMAX=compute_amax,
            num_warps=4,
            enable_fp_fusion=False,
        )
    amax = amax_i32.view(torch.float32) if compute_amax else None
    if amax is not None:
        register_amax(norm_out, amax)
    return h_out, norm_out, amax


# ---------------------------------------------------------------------------
# 3. Single-pass Q/K RMSNorm (width=256) + Partial RoPE (rotary_dim=64)
#    Reads strided [B, T, H, 256] directly, writes contiguous [B, T, H, 256].
# ---------------------------------------------------------------------------

@triton.jit
def _qk_norm_partial_rope_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    B_T,
    H,
    x_bt_stride,
    x_h_stride,
    cos_bt_stride,
    eps: tl.constexpr,
    block_heads: tl.constexpr,
):
    # Grid: (cdiv(H, block_heads), B_T)
    bt = tl.program_id(1).to(tl.int64)
    h = tl.program_id(0) * block_heads + tl.arange(0, block_heads)
    valid_h = h < H
    h_i64 = h[:, None].to(tl.int64)

    x_base = bt * x_bt_stride + h_i64 * x_h_stride
    out_base = (bt * H + h_i64) * 256

    col128 = tl.arange(0, 128)
    valid128 = valid_h[:, None]

    # Load all 256 elements into registers in two 128-wide chunks (exact ATen 4-stream reduction).
    v0 = tl.load(x_ptr + x_base + col128[None, :], valid128, other=0.0).to(tl.float32)
    v1 = tl.load(x_ptr + x_base + (col128[None, :] + 128), valid128, other=0.0).to(tl.float32)

    e0, o0 = tl.split(tl.reshape(v0 * v0, (block_heads, 32, 2, 2)))
    a0, c0 = tl.split(e0)
    b0, d0 = tl.split(o0)
    e1, o1 = tl.split(tl.reshape(v1 * v1, (block_heads, 32, 2, 2)))
    a1, c1 = tl.split(e1)
    b1, d1 = tl.split(o1)
    total = tl.sum((((a0 + a1) + (b0 + b1)) + (c0 + c1)) + (d0 + d1), axis=1)
    inv_rms = tl.rsqrt(total * (1.0 / 256.0) + eps)

    # Second half (dims 128..255): pass-through after RMSNorm.
    w1 = tl.load(weight_ptr + col128 + 128).to(tl.float32)
    y1_bf16 = ((v1 * inv_rms[:, None]) * (1.0 + w1[None, :])).to(tl.bfloat16)
    tl.store(out_ptr + out_base + (col128[None, :] + 128), y1_bf16, valid128)

    # First half (dims 0..127): split into 0..63 (rotary) and 64..127 (pass-through).
    w0 = tl.load(weight_ptr + col128).to(tl.float32)
    y0_bf16 = ((v0 * inv_rms[:, None]) * (1.0 + w0[None, :])).to(tl.bfloat16)

    # Store 64..127 pass-through directly; apply partial RoPE to 0..63.
    col64 = tl.arange(0, 64)
    # Extract 64..127 from y0_bf16? Store y0_bf16 first (or compute rotary directly from x_ptr/v0).
    # Storing 64..127 from y0_bf16 and overwriting 0..63 is 1 coalesced store + 1 rotary store,
    # or better: load 0..31 / 32..63 from v0 reshaped as (block_heads, 2, 64) -> split!
    # Triton tl.split splits the LAST dim of size 2; so reshape (block_heads, 4, 32) isn't split-able on dim 1.
    # Instead store y0_bf16 to [0..127] and compute rotary on 0..31 & 32..63 cleanly without extra global reads!
    # Wait: even cleaner — compute rotary from 4x 32-element loads? No, v0 is already in registers!
    # Storing y0_bf16 for 64..127 only: use mask col128 >= 64!
    tl.store(out_ptr + out_base + col128[None, :], y0_bf16, valid128 & (col128[None, :] >= 64))

    # Now compute dims 0..31 and 32..63 cleanly from x_ptr (already in L1 cache, 128 bytes!).
    col32 = tl.arange(0, 32)
    valid32 = valid_h[:, None]
    x_a = tl.load(x_ptr + x_base + col32[None, :], valid32, other=0.0).to(tl.float32)
    x_b = tl.load(x_ptr + x_base + (col32[None, :] + 32), valid32, other=0.0).to(tl.float32)
    w_a = tl.load(weight_ptr + col32).to(tl.float32)
    w_b = tl.load(weight_ptr + col32 + 32).to(tl.float32)

    q_a = ((x_a * inv_rms[:, None]) * (1.0 + w_a[None, :])).to(tl.bfloat16).to(tl.float32)
    q_b = ((x_b * inv_rms[:, None]) * (1.0 + w_b[None, :])).to(tl.bfloat16).to(tl.float32)

    cos_base = bt * cos_bt_stride
    c_a = tl.load(cos_ptr + cos_base + col32).to(tl.float32)[None, :]
    c_b = tl.load(cos_ptr + cos_base + col32 + 32).to(tl.float32)[None, :]
    s_a = tl.load(sin_ptr + cos_base + col32).to(tl.float32)[None, :]
    s_b = tl.load(sin_ptr + cos_base + col32 + 32).to(tl.float32)[None, :]

    # Match eager BF16: (q_rot * cos) + (rotate_half(q_rot) * sin)
    r_a = ((q_a * c_a).to(tl.bfloat16).to(tl.float32) + ((-q_b) * s_a).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    r_b = ((q_b * c_b).to(tl.bfloat16).to(tl.float32) + (q_a * s_b).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + out_base + col32[None, :], r_a, valid32)
    tl.store(out_ptr + out_base + (col32[None, :] + 32), r_b, valid32)


def fused_qk_norm_partial_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply exact Qwen3.5 RMSNorm (D=256) + partial RoPE (rotary_dim=64) in one pass.

    Accepts ``x`` of shape ``[B, T, H, 256]`` (may be strided along ``H``, e.g.,
    chunked from ``q_proj``) and ``cos, sin`` of shape ``[B, T, 64]``.
    Returns contiguous ``[B, T, H, 256]`` in token-major layout for FlashAttention.
    """
    B, T, H, D = x.shape
    rotary_dim = cos.shape[-1]
    if (
        not x.is_cuda
        or D != 256
        or rotary_dim != 64
        or x.dtype != weight.dtype == cos.dtype == sin.dtype == torch.bfloat16
        or x.stride(3) != 1
        or not weight.is_contiguous()
        or cos.shape != (B, T, 64)
        or sin.shape != (B, T, 64)
        or not (cos.is_contiguous() and sin.is_contiguous())
        or (B > 1 and x.stride(0) != T * x.stride(1))
    ):
        # Exact eager fallback
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)
        c = cos.unsqueeze(2)
        s = sin.unsqueeze(2)
        yr, yp = y[..., :rotary_dim], y[..., rotary_dim:]
        half = rotary_dim // 2
        rot_half = torch.cat((-yr[..., half:], yr[..., :half]), dim=-1)
        y_rot = (yr * c) + (rot_half * s)
        return torch.cat((y_rot, yp), dim=-1).contiguous()

    out = torch.empty((B, T, H, D), dtype=x.dtype, device=x.device)
    bt = B * T
    if bt == 0:
        return out
    if bt * H < 16:
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)
        c = cos.unsqueeze(2)
        s = sin.unsqueeze(2)
        yr, yp = y[..., :rotary_dim], y[..., rotary_dim:]
        half = rotary_dim // 2
        rot_half = torch.cat((-yr[..., half:], yr[..., :half]), dim=-1)
        y_rot = (yr * c) + (rot_half * s)
        return torch.cat((y_rot, yp), dim=-1).contiguous()
    block_heads = 4
    with torch.cuda.device(x.device):
        _qk_norm_partial_rope_kernel[(triton.cdiv(H, block_heads), bt)](
            x,
            weight,
            cos,
            sin,
            out,
            bt,
            H,
            x.stride(1),
            x.stride(2),
            cos.stride(1),
            eps=eps,
            block_heads=block_heads,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


# ---------------------------------------------------------------------------
# 4. Fused Full-Attention Output Gate: out = attn * sigmoid(gate) + inline amax
#    Reads contiguous attn [B, T, H, 256] and strided gate [B, T, H, 256].
# ---------------------------------------------------------------------------

@triton.jit
def _attn_sigmoid_gate_amax_kernel(
    attn_ptr,
    gate_ptr,
    out_ptr,
    amax_i32_ptr,
    B_T,
    H,
    gate_bt_stride,
    gate_h_stride,
    block_heads: tl.constexpr,
):
    bt = tl.program_id(1).to(tl.int64)
    h = tl.program_id(0) * block_heads + tl.arange(0, block_heads)
    valid_h = h < H
    h_i64 = h[:, None].to(tl.int64)

    attn_base = (bt * H + h_i64) * 256
    gate_base = bt * gate_bt_stride + h_i64 * gate_h_stride

    col = tl.arange(0, 256)
    mask = valid_h[:, None]

    a = tl.load(attn_ptr + attn_base + col[None, :], mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + gate_base + col[None, :], mask, other=0.0).to(tl.float32)
    sig_bf16 = tl.div_rn(1.0, 1.0 + libdevice.exp(-g)).to(tl.bfloat16)
    y_bf16 = (a * sig_bf16.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + attn_base + col[None, :], y_bf16, mask)

    y_f32 = tl.where(mask, y_bf16.to(tl.float32), 0.0)
    block_max = tl.max(_abs_f32_to_sortable_i32(y_f32))
    tl.atomic_max(amax_i32_ptr, block_max)


def fused_attn_sigmoid_gate_amax(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``attn_out.reshape(B, T, -1) * sigmoid(gate)`` + inline FP32 ``amax``.

    ``attn_out`` is contiguous ``[B, T, H, 256]``; ``gate`` is ``[B, T, H, 256]``
    and may be non-contiguous in ``H`` (sliced from ``q_proj``).
    """
    B, T, H, D = attn_out.shape
    if (
        not attn_out.is_cuda
        or D != 256
        or attn_out.dtype != gate.dtype == torch.bfloat16
        or attn_out.shape != gate.shape
        or not attn_out.is_contiguous()
        or gate.stride(3) != 1
        or (B > 1 and gate.stride(0) != T * gate.stride(1))
    ):
        out = attn_out.reshape(B, T, -1).contiguous() * torch.sigmoid(gate.reshape(B, T, -1))
        amax = torch.linalg.vector_norm(out, ord=float("inf")).float().reshape(1)
        register_amax(out, amax)
        return out, amax

    out = torch.empty((B, T, H * D), dtype=attn_out.dtype, device=attn_out.device)
    bt = B * T
    if bt == 0:
        return out, torch.zeros(1, dtype=torch.float32, device=attn_out.device)

    amax_i32 = torch.zeros(1, dtype=torch.int32, device=attn_out.device)
    block_heads = 4
    with torch.cuda.device(attn_out.device):
        _attn_sigmoid_gate_amax_kernel[(triton.cdiv(H, block_heads), bt)](
            attn_out,
            gate,
            out,
            amax_i32,
            bt,
            H,
            gate.stride(1),
            gate.stride(2),
            block_heads=block_heads,
            num_warps=4,
            enable_fp_fusion=False,
        )
    amax = amax_i32.view(torch.float32)
    register_amax(out, amax)
    return out, amax


# ---------------------------------------------------------------------------
# 5. Fused DeltaNet Gated RMSNorm (D=128) + inline amax
# ---------------------------------------------------------------------------

@triton.jit
def _gated_rms_norm_amax_kernel(
    hidden_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    amax_i32_ptr,
    rows,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, 128)
    offset = row[:, None].to(tl.int64) * 128 + column[None, :]
    valid = row[:, None] < rows
    hidden = tl.load(hidden_ptr + offset, valid, other=0.0).to(tl.float32)
    square = hidden * hidden

    even, odd = tl.split(tl.reshape(square, (block_rows, 32, 2, 2)))
    first, third = tl.split(even)
    second, fourth = tl.split(odd)
    total = tl.sum(((first + second) + third) + fourth, axis=1)
    inv_rms = tl.rsqrt(total * (1.0 / 128.0) + eps)

    normed = (hidden * inv_rms[:, None]).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(weight_ptr + column).to(tl.float32)
    weighted = (normed * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(gate_ptr + offset, valid, other=0.0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    y_bf16 = (weighted * silu).to(tl.bfloat16)
    tl.store(output_ptr + offset, y_bf16, valid)

    y_f32 = tl.where(valid, y_bf16.to(tl.float32), 0.0)
    block_max = tl.max(_abs_f32_to_sortable_i32(y_f32))
    tl.atomic_max(amax_i32_ptr, block_max)


def fused_gated_rms_norm_amax(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_shape: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Qwen3.5 gated RMSNorm (D=128) + inline FP32 ``amax``."""
    h2 = hidden.reshape(-1, 128)
    g2 = gate.reshape(-1, 128)
    rows = h2.shape[0]
    shape = out_shape or tuple(hidden.shape)
    if (
        not h2.is_cuda
        or h2.dtype != g2.dtype == weight.dtype == torch.bfloat16
        or h2.shape != g2.shape
        or weight.shape != (128,)
        or rows < 16
        or not (h2.is_contiguous() and g2.is_contiguous() and weight.is_contiguous())
    ):
        hf = h2.float()
        var = hf.pow(2).mean(-1, keepdim=True)
        normed = (hf * torch.rsqrt(var + eps)).to(h2.dtype)
        out = ((weight * normed) * torch.nn.functional.silu(g2.float())).to(h2.dtype).reshape(shape)
        amax = torch.linalg.vector_norm(out, ord=float("inf")).float().reshape(1)
        register_amax(out, amax)
        return out, amax

    out = torch.empty(shape, dtype=h2.dtype, device=h2.device)
    amax_i32 = torch.zeros(1, dtype=torch.int32, device=h2.device)
    block_rows = 4
    with torch.cuda.device(h2.device):
        _gated_rms_norm_amax_kernel[(triton.cdiv(rows, block_rows),)](
            h2, g2, weight, out, amax_i32, rows, eps, block_rows, num_warps=4, enable_fp_fusion=False,
        )
    amax = amax_i32.view(torch.float32)
    register_amax(out, amax)
    return out, amax


# ---------------------------------------------------------------------------
# 6. Token-major Isolated / Plain Causal Depthwise Conv (k=4) + SiLU
#    Supports arbitrary predecessor indices (for isolated question restarts &
#    left-padded multi-sequence batches) in a single pass with zero transposes.
# ---------------------------------------------------------------------------

@triton.jit
def _isolated_conv4_silu_kernel(
    x_ptr,
    weight_ptr,
    prev_ptr,
    out_ptr,
    total_tokens,
    channels: tl.constexpr,
    block_tokens: tl.constexpr,
    block_channels: tl.constexpr,
):
    c = tl.program_id(0) * block_channels + tl.arange(0, block_channels)
    t = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    t_valid = t < total_tokens
    c_valid = c < channels
    out_mask = t_valid[:, None] & c_valid[None, :]
    c_i64 = c[None, :].to(tl.int64)

    # Load all 4 weights for this channel tile once.
    w0 = tl.load(weight_ptr + c * 4 + 0, c_valid, other=0.0).to(tl.float32)
    w1 = tl.load(weight_ptr + c * 4 + 1, c_valid, other=0.0).to(tl.float32)
    w2 = tl.load(weight_ptr + c * 4 + 2, c_valid, other=0.0).to(tl.float32)
    w3 = tl.load(weight_ptr + c * 4 + 3, c_valid, other=0.0).to(tl.float32)

    # Predecessor indices for taps t-3, t-2, t-1 stored as prev[0]=t-1, prev[1]=t-2, prev[2]=t-3.
    # For tap 3 (current token t), prev[3, t] = t (or -1 if padding).
    p0 = tl.load(prev_ptr + 2 * total_tokens + t, t_valid, other=-1).to(tl.int64)
    p1 = tl.load(prev_ptr + 1 * total_tokens + t, t_valid, other=-1).to(tl.int64)
    p2 = tl.load(prev_ptr + 0 * total_tokens + t, t_valid, other=-1).to(tl.int64)
    p3 = tl.load(prev_ptr + 3 * total_tokens + t, t_valid, other=-1).to(tl.int64)

    x0 = tl.load(x_ptr + p0[:, None] * channels + c_i64, out_mask & (p0[:, None] >= 0), other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + p1[:, None] * channels + c_i64, out_mask & (p1[:, None] >= 0), other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + p2[:, None] * channels + c_i64, out_mask & (p2[:, None] >= 0), other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + p3[:, None] * channels + c_i64, out_mask & (p3[:, None] >= 0), other=0.0).to(tl.float32)

    acc = (((x0 * w0[None, :]) + (x1 * w1[None, :])) + (x2 * w2[None, :])) + (x3 * w3[None, :])
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    activated = tl.div_rn(rounded, 1.0 + libdevice.exp(-rounded)).to(tl.bfloat16)
    tl.store(out_ptr + t[:, None].to(tl.int64) * channels + c_i64, activated, out_mask)


@triton.jit
def _isolated_conv4_silu_split_kernel(
    x_ptr,
    weight_ptr,
    prev_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    total_tokens,
    channels: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    block_tokens: tl.constexpr,
    block_channels: tl.constexpr,
):
    c_pid = tl.program_id(0)
    c0 = c_pid * block_channels
    c = c0 + tl.arange(0, block_channels)
    t = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    t_valid = t < total_tokens
    c_valid = c < channels
    out_mask = t_valid[:, None] & c_valid[None, :]
    c_i64 = c[None, :].to(tl.int64)

    w0 = tl.load(weight_ptr + c * 4 + 0, c_valid, other=0.0).to(tl.float32)
    w1 = tl.load(weight_ptr + c * 4 + 1, c_valid, other=0.0).to(tl.float32)
    w2 = tl.load(weight_ptr + c * 4 + 2, c_valid, other=0.0).to(tl.float32)
    w3 = tl.load(weight_ptr + c * 4 + 3, c_valid, other=0.0).to(tl.float32)

    p0 = tl.load(prev_ptr + 2 * total_tokens + t, t_valid, other=-1).to(tl.int64)
    p1 = tl.load(prev_ptr + 1 * total_tokens + t, t_valid, other=-1).to(tl.int64)
    p2 = tl.load(prev_ptr + 0 * total_tokens + t, t_valid, other=-1).to(tl.int64)
    p3 = tl.load(prev_ptr + 3 * total_tokens + t, t_valid, other=-1).to(tl.int64)

    x0 = tl.load(x_ptr + p0[:, None] * channels + c_i64, out_mask & (p0[:, None] >= 0), other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + p1[:, None] * channels + c_i64, out_mask & (p1[:, None] >= 0), other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + p2[:, None] * channels + c_i64, out_mask & (p2[:, None] >= 0), other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + p3[:, None] * channels + c_i64, out_mask & (p3[:, None] >= 0), other=0.0).to(tl.float32)

    acc = (((x0 * w0[None, :]) + (x1 * w1[None, :])) + (x2 * w2[None, :])) + (x3 * w3[None, :])
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    activated = tl.div_rn(rounded, 1.0 + libdevice.exp(-rounded)).to(tl.bfloat16)

    # Write directly into contiguous Q [T, key_dim], K [T, key_dim], or V [T, value_dim]
    # so FLA chunk_gated_delta_rule avoids 3x per-layer .contiguous() copies.
    t_i64 = t[:, None].to(tl.int64)
    if c0 < key_dim:
        tl.store(q_out_ptr + t_i64 * key_dim + c_i64, activated, out_mask)
    elif c0 < 2 * key_dim:
        tl.store(k_out_ptr + t_i64 * key_dim + (c_i64 - key_dim), activated, out_mask)
    else:
        tl.store(v_out_ptr + t_i64 * value_dim + (c_i64 - 2 * key_dim), activated, out_mask)


def fused_isolated_conv4_silu_split(
    x: torch.Tensor,
    weight: torch.Tensor,
    prev: torch.Tensor,
    key_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused conv4+SiLU writing contiguous ``(q, k, v)`` directly (zero FLA copies)."""
    B, T, C = x.shape
    block_channels = 128
    if C != 2 * key_dim + value_dim or (key_dim % block_channels) != 0 or (value_dim % block_channels) != 0:
        mixed = fused_isolated_conv4_silu(x, weight, prev)
        q, k, v = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
        return q.contiguous(), k.contiguous(), v.contiguous()

    total_tokens = B * T
    q_out = torch.empty((B, T, key_dim), dtype=x.dtype, device=x.device)
    k_out = torch.empty((B, T, key_dim), dtype=x.dtype, device=x.device)
    v_out = torch.empty((B, T, value_dim), dtype=x.dtype, device=x.device)
    if total_tokens == 0:
        return q_out, k_out, v_out
    block_tokens = 8
    with torch.cuda.device(x.device):
        _isolated_conv4_silu_split_kernel[
            (triton.cdiv(C, block_channels), triton.cdiv(total_tokens, block_tokens))
        ](
            x,
            weight,
            prev,
            q_out,
            k_out,
            v_out,
            total_tokens,
            channels=C,
            key_dim=key_dim,
            value_dim=value_dim,
            block_tokens=block_tokens,
            block_channels=block_channels,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out, v_out


def fused_isolated_conv4_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    prev: torch.Tensor,
) -> torch.Tensor:
    """Token-major causal depthwise conv (k=4) + SiLU with predecessor table.

    Args:
        x: contiguous BF16 ``[B, T, C]``.
        weight: contiguous BF16 ``[C, 4]`` (``conv1d.weight.squeeze(1)``).
        prev: contiguous int32 ``[4, B * T]`` where rows 0, 1, 2 hold flat source
            indices for ``t-1, t-2, t-3`` and row 3 holds ``t`` (``-1`` for pad).
    Returns:
        contiguous BF16 ``[B, T, C]``.
    """
    B, T, C = x.shape
    total_tokens = B * T
    out = torch.empty_like(x)
    if total_tokens == 0:
        return out
    block_tokens, block_channels = 8, 128
    with torch.cuda.device(x.device):
        _isolated_conv4_silu_kernel[
            (triton.cdiv(C, block_channels), triton.cdiv(total_tokens, block_tokens))
        ](
            x,
            weight,
            prev,
            out,
            total_tokens,
            channels=C,
            block_tokens=block_tokens,
            block_channels=block_channels,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
