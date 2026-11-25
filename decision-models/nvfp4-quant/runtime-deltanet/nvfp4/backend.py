"""FlashInfer adapter: activation quantisation, backend policy and the FP4 GEMM call.

Backend policy (explicit, never ``auto``)
-----------------------------------------
* SM121 (GB10) / SM120: ``b12x`` -- FlashInfer's native SM120/SM121 warp-level
  block-scaled FP4 kernel.  ``cute-dsl`` is *refused* by FlashInfer on capability
  121 (``BackendSupportedError``), so ``cute-dsl`` is accepted as an alias and
  resolved to ``b12x`` on 12.x with a warning.
* SM100/SM103/SM107: ``cute-dsl``.
* ``cutlass`` is kept for diagnostics only; ``auto`` is rejected because it does
  not guarantee a native FP4 backend.

The activation quantiser is ``flashinfer.nvfp4_quantize(backend="cuda")`` with
one FP32 per-tensor global scale and per-16 block scales in the 128x4 swizzled
layout, matching the checkpoint's weight recipe (Max PTQ).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import re
from typing import Any

import torch

logger = logging.getLogger("nvfp4.backend")

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
AMAX_SCALE = FP4_MAX * FP8_E4M3_MAX  # 2688.0
BLOCK_SIZE = 16

#: Backends accepted by this adapter.  ``auto`` is deliberately absent.
BACKENDS = ("b12x", "cute-dsl", "cutlass")

_MIN_FLASHINFER = (0, 6, 18)

#: Measured on GB10: below these M, b12x is at least as fast as cutlass; at or
#: above them, cutlass is materially faster (the b12x default plan collapses for
#: the narrow-N / large-K down_proj).  Both backends are native block-scaled FP4
#: and were verified bit-identical on the real weights.  See
#: ``reports/backend_sweep.json``.
CUTLASS_MIN_M = {
    (2560, 9216): 4096,  # mlp.down_proj
    (2560, 4096): 7000,  # attn.o_proj / linear_attn.out_proj
    (8192, 2560): 7000,  # linear_attn.in_proj_qkv / attn.q_proj
}


def policy_enabled() -> bool:
    """The per-shape policy is on unless a single backend is forced explicitly."""
    return (
        os.environ.get("DECIDER_NVFP4_BACKEND") in (None, "")
        and os.environ.get("DECIDER_NVFP4_POLICY", "1") != "0"
    )


def backend_for(n: int, k: int, m: int, base: str) -> str:
    """Backend for one GEMM: measured cutlass crossover, else the base backend."""
    if not policy_enabled():
        return base
    threshold = CUTLASS_MIN_M.get((int(n), int(k)))
    if threshold is not None and m >= threshold:
        return "cutlass"
    return base


def _flashinfer_version() -> tuple[int, ...]:
    import flashinfer

    raw = getattr(flashinfer, "__version__", "0")
    parts = re.findall(r"\d+", raw)[:3]
    return tuple(int(p) for p in parts) if parts else (0,)


def _capability(device: torch.device | int | None = None) -> tuple[int, int]:
    if device is None:
        device = torch.cuda.current_device()
    if isinstance(device, torch.device):
        device = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(int(device))


def _pdl_default() -> bool:
    """Programmatic Dependent Launch is off by default on SM12x (see report)."""
    return os.environ.get("DECIDER_NVFP4_PDL", "0") == "1"


def resolve_backend(requested: str | None = None, device: torch.device | int | None = None) -> str:
    """Return the concrete FlashInfer backend for ``device``.

    ``requested`` defaults to ``$DECIDER_NVFP4_BACKEND`` and then ``b12x``.
    Raises ``ValueError`` for an unsupported explicit backend; it never falls
    back silently to a different precision path.
    """
    requested = (requested or os.environ.get("DECIDER_NVFP4_BACKEND") or "b12x").strip().lower()
    if requested == "auto":
        raise ValueError(
            "DECIDER_NVFP4_BACKEND=auto is forbidden: it does not guarantee a native FP4 backend. "
            f"Use one of {', '.join(BACKENDS)}."
        )
    if requested not in BACKENDS:
        raise ValueError(f"Unknown NVFP4 backend {requested!r}; expected one of {BACKENDS}")

    major, minor = _capability(device)
    cap = major * 10 + minor

    if requested == "b12x":
        if cap not in (120, 121):
            raise ValueError(f"b12x requires SM120/SM121, this device is SM{cap}.")
        if _flashinfer_version() < _MIN_FLASHINFER:
            raise ValueError(
                f"b12x requires FlashInfer >= {'.'.join(map(str, _MIN_FLASHINFER))} "
                f"(installed {_flashinfer_version()}); the 0.6.3 cutlass kernel fails on SM121."
            )
        return "b12x"

    if requested == "cute-dsl":
        if cap in (120, 121):
            logger.warning(
                "cute-dsl is not supported on SM%d%d (FlashInfer refuses it); using native b12x instead.",
                major, minor,
            )
            return resolve_backend("b12x", device)
        if cap not in (100, 103, 107):
            raise ValueError(f"cute-dsl requires SM100/SM103/SM107, this device is SM{cap}.")
        return "cute-dsl"

    # cutlass: diagnostics/comparison only.  Requires the same 128x4 inputs.
    if cap not in (100, 103, 110, 120, 121):
        raise ValueError(f"cutlass FP4 GEMM is not supported on SM{cap}.")
    return "cutlass"


def quantize_activation(
    x: torch.Tensor,
    *,
    enable_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise a contiguous BF16 ``[M, K]`` activation block to NVFP4.

    Returns ``(packed uint8 [M, K/2], block_scale uint8 [PR*PC], global_scale fp32 [1])``
    where ``PR = ceil(M/128)*128`` and ``PC = ceil((K/16)/4)*4``.

    The global scale is the quantisation multiplier ``G_x = 2688 / amax(x)``
    (all-zero rows use 1).  Non-finite inputs raise instead of being silently
    replaced; the check is device-side so the hot path does not synchronise.
    """
    from flashinfer import SfLayout, nvfp4_quantize

    if x.dim() != 2:
        raise ValueError(f"quantize_activation expects [M, K], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"quantize_activation expects bfloat16, got {x.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()

    # Infinity norm == max(abs(x)); PyTorch's fused norm reduction is ~3x faster
    # than abs().amax() on GB10 and avoids the full-size temporary (measured in
    # scripts/probe_amax.py; values are identical).
    amax = torch.linalg.vector_norm(x, ord=float("inf")).float().reshape(1)
    # Device-side assert: no .item()/sync in the hot path, no silent NaN repair.
    torch._assert(torch.isfinite(amax).all(), "NVFP4 activation contains non-finite values")
    is_zero = amax == 0
    safe_amax = torch.where(is_zero, torch.ones_like(amax), amax)
    g_x = torch.where(is_zero, torch.ones_like(amax), AMAX_SCALE / safe_amax)

    if enable_pdl is None:
        enable_pdl = _pdl_default()

    x_q, x_sf = nvfp4_quantize(
        x,
        g_x,
        sfLayout=SfLayout.layout_128x4,
        sf_vec_size=BLOCK_SIZE,
        do_shuffle=False,
        backend="cuda",
        per_token_activation=False,
        enable_pdl=enable_pdl,
    )
    return x_q.view(torch.uint8), x_sf.view(torch.uint8), g_x


# --------------------------------------------------------------------------
# Forward-scoped activation-quantisation reuse.
# Several projections consume the exact same BF16 tensor object (qkv/z/a/b,
# q/k/v, gate/up).  Reusing the packed activation is bit-identical; the cache is
# keyed by tensor identity and lives only for one model forward.
# --------------------------------------------------------------------------
_QUANT_CACHE: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "nvfp4_quant_cache", default=None
)


@contextlib.contextmanager
def quant_cache_scope(*, enabled: bool | None = None):
    if enabled is None:
        enabled = os.environ.get("DECIDER_NVFP4_SHARED_QUANT", "0") == "1"
    token = _QUANT_CACHE.set({} if enabled else None)
    try:
        yield
    finally:
        _QUANT_CACHE.reset(token)


def cached_quantize(key: torch.Tensor, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``quantize_activation(flat)``, reused when ``key`` was seen in this scope."""
    cache = _QUANT_CACHE.get()
    if cache is None:
        return quantize_activation(flat)
    entry = cache.get(id(key))
    if entry is not None and entry[0] is key and entry[1] == flat.shape:
        return entry[2]
    value = quantize_activation(flat)
    cache[id(key)] = (key, flat.shape, value)
    return value


def mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    backend: str,
    out_dtype: torch.dtype = torch.bfloat16,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    """Native FP4 GEMM ``Y = alpha * (A_fp4 @ B_fp4)`` with explicit backend."""
    from flashinfer.gemm import mm_fp4 as _mm_fp4

    if enable_pdl is None:
        enable_pdl = _pdl_default()
    try:
        return _mm_fp4(
            a,
            b,
            a_scale,
            b_scale,
            alpha=alpha,
            out_dtype=out_dtype,
            block_size=BLOCK_SIZE,
            use_8x4_sf_layout=False,
            backend=backend,
            use_nvfp4=True,
            enable_pdl=enable_pdl,
        )
    except Exception as exc:  # keep the shape context; never fall back
        raise RuntimeError(
            f"FlashInfer mm_fp4 failed (backend={backend}, A={tuple(a.shape)}, B={tuple(b.shape)}, "
            f"M={a.shape[0]}, N={b.shape[1]}, K={a.shape[1] * 2}): {type(exc).__name__}: {exc}"
        ) from exc


def backend_report(device: torch.device | str = "cuda") -> dict[str, Any]:
    """Environment / backend facts for the runtime report."""
    import flashinfer

    dev = torch.device(device)
    major, minor = _capability(dev)
    resolved = resolve_backend(device=dev)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "device": torch.cuda.get_device_name(dev),
        "capability": [major, minor],
        "backend_requested": os.environ.get("DECIDER_NVFP4_BACKEND", "b12x"),
        "backend": resolved,
        "backend_policy": {
            "enabled": policy_enabled(),
            "cutlass_min_m": {f"{n}x{k}": m for (n, k), m in CUTLASS_MIN_M.items()},
        },
        "activation_quantizer": "flashinfer.nvfp4_quantize(backend='cuda', per_token_activation=False)",
        "gemm": "flashinfer.gemm.mm_fp4(use_nvfp4=True, block_size=16, use_8x4_sf_layout=False)",
        "enable_pdl": _pdl_default(),
        "amax_scale": AMAX_SCALE,
    }
