"""Online FP8 W8A8 Linear (E4M3FN) with fused Triton activation quantisation.

Supports two scaling modes on SM120/SM121 (GB10):
  * ``tensor``  : per-tensor weight & activation scale (``torch._scaled_mm`` runs
                  at ~183 TFLOPS across all shapes including ``down_proj``, and
                  directly reuses the inline FP32 ``amax`` computed by
                  ``fused_ops`` producer kernels with zero standalone absmax).
  * ``rowwise`` : per-output-channel weight scale + per-token activation scale
                  via a single-CTA fused Triton rowwise quantiser.

Both modes share layer-scoped quantized activations across projections that
consume the same input tensor (``qkv/z``, ``q/k/v``, ``gate/up``), eliminating
~36% of activation quantisation work.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any

import torch
from torch import nn
import triton
import triton.language as tl

if __package__:
    from .fused_ops import lookup_amax
else:
    from fused_ops import lookup_amax

FP8_E4M3_MAX = 448.0
_MIN_SCALE = 1e-12

# Layer-scoped FP8 activation quantisation cache: id(tensor) -> (tensor, flat_shape, (x_fp8, x_scale))
_FP8_QUANT_CACHE: contextvars.ContextVar[
    dict[int, tuple[torch.Tensor, tuple[int, ...], tuple[torch.Tensor, torch.Tensor]]] | None
] = contextvars.ContextVar("nvfp4_fp8_quant_cache", default=None)


@contextlib.contextmanager
def fp8_quant_cache_scope(*, enabled: bool = True):
    token = _FP8_QUANT_CACHE.set({} if enabled else None)
    try:
        yield
    finally:
        _FP8_QUANT_CACHE.reset(token)


# ---------------------------------------------------------------------------
# Fused Triton FP8 E4M3FN quantisers (zero FP32 HBM intermediates)
# ---------------------------------------------------------------------------

@triton.jit
def _quantize_fp8_tensor_kernel(
    x_ptr,
    amax_ptr,
    out_ptr,
    scale_out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    amax = tl.load(amax_ptr).to(tl.float32)
    safe_amax = tl.maximum(amax, 1e-12)
    scale = safe_amax * (1.0 / 448.0)
    inv_scale = 448.0 / safe_amax
    if pid == 0:
        tl.store(scale_out_ptr, scale)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.minimum(tl.maximum(x * inv_scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(out_ptr + offs, q, mask=mask)


@triton.jit
def _quantize_fp8_rowwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    rows,
    cols,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= rows:
        return
    base = row * cols

    row_max = tl.full((), 0.0, tl.float32)
    for k0 in range(0, cols, BLOCK_K):
        col = k0 + tl.arange(0, BLOCK_K)
        mask = col < cols
        vals = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(tl.abs(vals), axis=0))

    safe_max = tl.maximum(row_max, 1e-12)
    scale = safe_max * (1.0 / 448.0)
    inv_scale = 448.0 / safe_max
    tl.store(scale_ptr + row, scale)

    for k0 in range(0, cols, BLOCK_K):
        col = k0 + tl.arange(0, BLOCK_K)
        mask = col < cols
        vals = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(vals * inv_scale, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(out_ptr + base + col, q, mask=mask)


def quantize_fp8_tensor(
    flat: torch.Tensor,
    *,
    precomputed_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise contiguous BF16 ``[M, K]`` to per-tensor ``(float8_e4m3fn, fp32 [1])``."""
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if precomputed_amax is None:
        amax = torch.linalg.vector_norm(flat, ord=float("inf")).float().reshape(1)
    else:
        amax = precomputed_amax.reshape(1)

    n = flat.numel()
    out = torch.empty(flat.shape, dtype=torch.float8_e4m3fn, device=flat.device)
    scale = torch.empty(1, dtype=torch.float32, device=flat.device)
    if n == 0:
        scale.fill_(1.0)
        return out, scale

    if flat.is_cuda and flat.dtype == torch.bfloat16:
        block = 2048
        with torch.cuda.device(flat.device):
            _quantize_fp8_tensor_kernel[(triton.cdiv(n, block),)](
                flat, amax, out, scale, n, BLOCK=block, num_warps=8,
            )
        return out, scale

    safe = amax.clamp(min=_MIN_SCALE)
    scale.copy_(safe / FP8_E4M3_MAX)
    out.copy_((flat.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn))
    return out, scale


def quantize_fp8_rowwise(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise contiguous BF16 ``[M, K]`` to per-row ``(float8_e4m3fn, fp32 [M, 1])``."""
    if not flat.is_contiguous():
        flat = flat.contiguous()
    rows, cols = flat.shape
    out = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=flat.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=flat.device)
    if rows == 0 or cols == 0:
        scale.fill_(1.0)
        return out, scale

    if flat.is_cuda and flat.dtype == torch.bfloat16:
        block_k = 1024 if cols >= 1024 else triton.next_power_of_2(cols)
        with torch.cuda.device(flat.device):
            _quantize_fp8_rowwise_kernel[(rows,)](
                flat, out, scale, rows, cols, BLOCK_K=block_k, num_warps=8 if block_k >= 1024 else 4,
            )
        return out, scale

    amax = flat.float().abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE)
    scale.copy_(amax / FP8_E4M3_MAX)
    out.copy_((flat.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn))
    return out, scale


# ---------------------------------------------------------------------------
# FP8Linear module + backbone installer
# ---------------------------------------------------------------------------

class FP8Linear(nn.Linear):
    """Drop-in ``nn.Linear`` replacement backed by ``torch._scaled_mm`` (E4M3FN)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        scaling: str = "tensor",
        use_fast_accum: bool = False,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if scaling not in {"tensor", "rowwise"}:
            raise ValueError(f"Unknown FP8 scaling mode {scaling!r}; expected 'tensor' or 'rowwise'")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scaling = scaling
        self.use_fast_accum = bool(use_fast_accum)
        self.prefix = prefix
        self.register_buffer("weight_fp8", weight_fp8.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        self.register_buffer("bias", bias)
        self._w_t: torch.Tensor | None = None
        self._w_t_ptr: int = 0

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        scaling: str = "tensor",
        use_fast_accum: bool = False,
        prefix: str = "",
    ) -> "FP8Linear":
        w = linear.weight.detach()
        wf = w.float()
        if scaling == "rowwise":
            scale_col = wf.abs().amax(dim=1, keepdim=True).clamp(min=_MIN_SCALE) / FP8_E4M3_MAX
            wq = (wf / scale_col).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            w_scale = scale_col.t().contiguous()  # [1, N]
        elif scaling == "tensor":
            w_scale = (wf.abs().amax().clamp(min=_MIN_SCALE) / FP8_E4M3_MAX).reshape(1)
            wq = (wf / w_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        else:
            raise ValueError(f"Unknown FP8 scaling mode {scaling!r}; expected 'tensor' or 'rowwise'")

        bias = linear.bias.detach().to(torch.bfloat16).contiguous() if linear.bias is not None else None
        return cls(
            linear.in_features,
            linear.out_features,
            wq,
            w_scale,
            bias,
            scaling=scaling,
            use_fast_accum=use_fast_accum,
            prefix=prefix,
        )

    @property
    def weight(self) -> torch.Tensor:  # pragma: no cover - guard only
        raise RuntimeError(
            f"{self.prefix or 'FP8Linear'}: weight is stored as float8_e4m3fn (weight_fp8 / weight_scale)"
        )

    def _weight_t(self) -> torch.Tensor:
        ptr = self.weight_fp8.data_ptr()
        if self._w_t is None or self._w_t_ptr != ptr:
            self._w_t = self.weight_fp8.t()
            self._w_t_ptr = ptr
        return self._w_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()

        cache = _FP8_QUANT_CACHE.get()
        cached = None
        if cache is not None:
            entry = cache.get(id(x))
            if entry is not None and entry[0] is x and entry[1] == flat.shape:
                cached = entry[2]

        if cached is not None:
            x_fp8, x_scale = cached
        else:
            if self.scaling == "tensor":
                amax = lookup_amax(x, (flat.shape[0], flat.shape[1]))
                x_fp8, x_scale = quantize_fp8_tensor(flat, precomputed_amax=amax)
            else:
                x_fp8, x_scale = quantize_fp8_rowwise(flat)
            if cache is not None:
                cache[id(x)] = (x, flat.shape, (x_fp8, x_scale))

        out = torch._scaled_mm(
            x_fp8,
            self._weight_t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=self.use_fast_accum,
        )
        return out.reshape(*orig_shape[:-1], self.out_features)


def _patch_locate_for_fp8() -> None:
    """Allow Locate's LoRA wrapper to inspect ``FP8Linear`` without touching ``.weight``."""
    try:
        from pplx_decider import locate as _locate
    except Exception:
        return
    if getattr(_locate, "_fp8_patched", False):
        return
    original = _locate.LoRALinear.__init__

    def init(self: Any, base: nn.Module, rank: int, alpha: float) -> None:
        if isinstance(base, FP8Linear):
            nn.Module.__init__(self)
            self.base = base
            self.scale = alpha / rank
            device = base.weight_fp8.device
            self.lora_a = nn.Parameter(
                torch.zeros(rank, base.in_features, device=device), requires_grad=False
            )
            self.lora_b = nn.Parameter(
                torch.zeros(base.out_features, rank, device=device), requires_grad=False
            )
            self.enabled = False
            return
        original(self, base, rank, alpha)

    _locate.LoRALinear.__init__ = init
    _locate._fp8_patched = True


def install_fp8_linears(
    backbone: nn.Module,
    *,
    scaling: str = "tensor",
    min_out_features: int = 256,
    use_fast_accum: bool = False,
) -> dict[str, Any]:
    """Convert large text-backbone ``nn.Linear`` layers in-place to ``FP8Linear``.

    Matches the NVFP4 quantisation scope (200 large MLP/Attention/DeltaNet
    projections under ``backbone.language_model.layers``), keeping tiny gates
    (``in_proj_a/b``), embeddings, norms, vision encoder, and readout in BF16.
    """
    _patch_locate_for_fp8()
    text_layers = backbone.language_model.layers
    replaced = 0
    kept: list[str] = []

    def _walk(module: nn.Module, prefix: str) -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if type(child) is nn.Linear:
                if (
                    child.in_features % 16 == 0
                    and child.out_features % 16 == 0
                    and child.out_features >= min_out_features
                ):
                    fp8_mod = FP8Linear.from_linear(
                        child, scaling=scaling, use_fast_accum=use_fast_accum, prefix=full
                    )
                    setattr(module, name, fp8_mod)
                    del child
                    replaced += 1
                else:
                    kept.append(full)
            else:
                _walk(child, full)

    _walk(text_layers, "layers")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "fp8_modules": replaced,
        "kept_bf16_linears": len(kept),
        "scaling": scaling,
        "use_fast_accum": use_fast_accum,
    }


def detect_fp8_scaling(backbone: nn.Module) -> str | None:
    """Return ``'tensor'`` or ``'rowwise'`` if ``backbone`` has ``FP8Linear`` modules, else ``None``."""
    for mod in backbone.language_model.layers.modules():
        if isinstance(mod, FP8Linear):
            return mod.scaling
    return None
