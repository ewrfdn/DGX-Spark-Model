"""FlashInfer GDN prefill adapter (SM121) for the Decider's Gated DeltaNet core.

Contract, verified against the model's captured inputs:
  * the model's ``g`` is log-decay; FlashInfer expects the linear forget gate,
    so pass ``exp(g)``;
  * FlashInfer's ``use_qk_l2norm_in_kernel`` flag is documented but not
    dispatched in 0.6.18, so q/k are normalised here exactly once;
  * the model repeats q/k to the value-head count (GVA 2x); FlashInfer supports
    native GVA, so take every other repeated head instead of materialising 32;
  * ``beta`` must be FP32; q/k/v stay BF16.

Only prefill (B=1, no cache) is supported here.  Unsupported cases raise, so the
caller can keep the explicit baseline path instead of silently changing math.
"""

from __future__ import annotations

import os

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-6, fp32: bool = False) -> torch.Tensor:
    # BF16 vector_norm accumulates internally in FP32 and is ~5x faster than
    # the eager float() + pow/mean/rsqrt formulation (see reports/l2norm_bench.json).
    # The fp32 mode reproduces the model/FLA l2norm exactly.
    if fp32:
        xf = x.float()
        return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def flashinfer_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    gva_ratio: int = 2,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as _fi_gdn

    if query.dim() != 4 or query.shape[0] != 1:
        raise NotImplementedError(
            f"FlashInfer GDN adapter supports prefill B=1 only, got {tuple(query.shape)}"
        )
    if cu_seqlens is not None:
        raise NotImplementedError("Explicit cu_seqlens is not supported by this adapter yet")
    if initial_state is not None and initial_state.shape[0] != 1:
        raise NotImplementedError("Only B=1 initial_state is supported")

    q, k, v = query[0], key[0], value[0]
    # GVA: the model's repeat_interleave(2) produced [h0,h0,h1,h1,...].  Use the
    # even heads as the 16 unique q/k heads; v keeps 32 heads.
    if gva_ratio > 1 and q.shape[1] == v.shape[1] and q.shape[1] % gva_ratio == 0:
        q = q[:, ::gva_ratio, :]
        k = k[:, ::gva_ratio, :]
    fp32_norm = os.environ.get("DECIDER_GDN_NORM", "bf16").lower() == "fp32"
    q = _l2norm(q.contiguous(), fp32=fp32_norm)
    k = _l2norm(k.contiguous(), fp32=fp32_norm)
    v = v.contiguous()

    if g is None or beta is None:
        raise ValueError("FlashInfer GDN requires g and beta")
    g_lin = g[0].exp().float().contiguous()
    beta_f = beta[0].float().contiguous()
    T = q.shape[0]
    cu = torch.tensor([0, T], dtype=torch.int32, device=q.device)

    state = None
    if initial_state is not None:
        state = initial_state.contiguous()
    out = _fi_gdn(
        q,
        k,
        v,
        g=g_lin,
        beta=beta_f,
        scale=scale,
        initial_state=state,
        output_final_state=output_final_state,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
        use_cp=False,
    )
    if isinstance(out, tuple):
        y, final = out
        return y.unsqueeze(0), final
    return out.unsqueeze(0), None


def install_flashinfer_gdn(gva_ratio: int = 2) -> None:
    """Monkeypatch the Transformers GDN core call in this process.

    Unsupported shapes (e.g. B>1 padded batches) fall back to the original FLA
    path only with an explicit warning and a counter, never silently.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if getattr(modeling, "_nvfp4_gdn_installed", False):
        return
    original = modeling.torch_chunk_gated_delta_rule
    state = {"fallbacks": 0, "calls": 0}

    def wrapper(query, key, value, g=None, beta=None, **kwargs):
        state["calls"] += 1
        try:
            out = flashinfer_gdn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                scale=kwargs.get("scale"),
                initial_state=kwargs.get("initial_state"),
                output_final_state=kwargs.get("output_final_state", False),
                cu_seqlens=kwargs.get("cu_seqlens"),
                gva_ratio=gva_ratio,
            )
            return out
        except NotImplementedError as exc:
            state["fallbacks"] += 1
            if state["fallbacks"] == 1:
                print(f"[nvfp4] GDN flashinfer unsupported ({exc}); using FLA baseline for this call", flush=True)
            return original(query, key, value, g=g, beta=beta, **kwargs)

    wrapper.nvfp4_fallbacks = lambda: state["fallbacks"]
    wrapper.nvfp4_calls = lambda: state["calls"]
    modeling.torch_chunk_gated_delta_rule = wrapper
    modeling._nvfp4_gdn_installed = True
