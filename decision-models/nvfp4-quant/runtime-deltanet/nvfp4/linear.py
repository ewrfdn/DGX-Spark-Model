"""``NVFP4Linear``: a Linear whose weight stays packed NVFP4 at all times.

The module inherits from ``nn.Linear`` only so that third-party code doing
``isinstance(module, nn.Linear)`` (the Decider's Locate adapter) still sees a
Linear.  ``nn.Linear.__init__`` is **not** called: there is no BF16 ``weight``
parameter.  Use ``weight_packed`` / ``weight_block_scale`` /
``weight_global_scale``.
"""

from __future__ import annotations

import torch
from torch import nn

from .backend import BLOCK_SIZE, backend_for, cached_quantize, mm_fp4, resolve_backend


def sf_padded_dims(n: int, k: int, block_size: int = BLOCK_SIZE) -> tuple[int, int]:
    """Padded (rows, cols-in-blocks) of the 128x4 swizzled block-scale matrix."""
    blocks = k // block_size
    return ((n + 127) // 128) * 128, ((blocks + 3) // 4) * 4


class NVFP4Linear(nn.Linear):
    """``Y = X @ W.T + bias`` with W stored as packed NVFP4 (E2M1 + FP8 scales).

    Inputs are BF16 ``[..., K]``; activations are quantised on the fly to NVFP4
    and multiplied by the packed weight with FlashInfer's native FP4 GEMM.
    Output is BF16 ``[..., N]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_packed: torch.Tensor,
        weight_block_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        backend: str | None = None,
        prefix: str = "",
    ) -> None:
        # Deliberately skip nn.Linear.__init__: no BF16 weight Parameter.
        nn.Module.__init__(self)
        n, k = int(out_features), int(in_features)
        if n % 64 or k % 64:
            raise ValueError(f"{prefix or 'NVFP4Linear'}: N={n}, K={k} must both be multiples of 64")
        self.in_features = k
        self.out_features = n
        self.prefix = prefix
        self.backend = resolve_backend(backend) if backend else resolve_backend()

        _validate_packed(prefix, weight_packed, n, k)
        _validate_scale(prefix, weight_block_scale, n, k)
        _validate_global_scale(prefix, weight_global_scale)
        if bias is not None:
            if tuple(bias.shape) != (n,):
                raise ValueError(f"{prefix}: bias shape {tuple(bias.shape)} != ({n},)")
            if bias.dtype != torch.bfloat16:
                raise ValueError(f"{prefix}: bias dtype {bias.dtype} != bfloat16")
            if bias.device != weight_packed.device:
                raise ValueError(f"{prefix}: bias is on {bias.device}, weight on {weight_packed.device}")

        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("weight_block_scale", weight_block_scale)
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer("bias", bias)
        self._pr, self._pc = sf_padded_dims(n, k)
        self._b_view: torch.Tensor | None = None
        self._b_view_key: tuple[int, int] | None = None
        self._b_scale_view: torch.Tensor | None = None
        self._b_scale_view_key: tuple[int, int] | None = None

    # ------------------------------------------------------------------ views

    @property
    def weight(self) -> torch.Tensor:  # pragma: no cover - guard only
        raise RuntimeError(
            f"{self.prefix or 'NVFP4Linear'}: the weight is packed (weight_packed/weight_block_scale); "
            "there is no BF16 weight matrix to read"
        )

    def _b(self) -> torch.Tensor:
        """B for ``mm_fp4``: [K/2, N], column-major view of the [N, K/2] packed weight."""
        key = (self.weight_packed.data_ptr(), self.weight_packed.device.index or 0)
        if self._b_view is None or self._b_view_key != key:
            self._b_view = self.weight_packed.t()
            self._b_view_key = key
        return self._b_view

    def _b_scale(self) -> torch.Tensor:
        """Block scales as [PC, PR] (the transpose is a view; FlashInfer reads the flat buffer)."""
        key = (self.weight_block_scale.data_ptr(), self.weight_block_scale.device.index or 0)
        if self._b_scale_view is None or self._b_scale_view_key != key:
            self._b_scale_view = self.weight_block_scale.reshape(self._pr, self._pc).t()
            self._b_scale_view_key = key
        return self._b_scale_view

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix or 'NVFP4Linear'}: input dtype {x.dtype} != bfloat16")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.prefix or 'NVFP4Linear'}: input K={x.shape[-1]} != {self.in_features}"
            )
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if x2.shape[0] == 0:
            return x.new_empty((*leading, self.out_features))

        x_q, x_sf, g_x = cached_quantize(x, x2)
        alpha = torch.reciprocal(g_x * self.weight_global_scale)
        y = mm_fp4(
            x_q,
            self._b(),
            x_sf,
            self._b_scale(),
            alpha,
            backend=backend_for(self.out_features, self.in_features, x2.shape[0], self.backend),
            out_dtype=torch.bfloat16,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*leading, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend}, nvfp4=packed"
        )


def _validate_packed(prefix: str, packed: torch.Tensor, n: int, k: int) -> None:
    if packed.dtype != torch.uint8:
        raise ValueError(f"{prefix}: weight_packed dtype {packed.dtype} != uint8")
    if tuple(packed.shape) != (n, k // 2):
        raise ValueError(f"{prefix}: weight_packed shape {tuple(packed.shape)} != ({n}, {k // 2})")
    if not packed.is_contiguous():
        raise ValueError(f"{prefix}: weight_packed is not contiguous")


def _validate_scale(prefix: str, scale: torch.Tensor, n: int, k: int) -> None:
    if scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError(f"{prefix}: block-scale dtype {scale.dtype} is not uint8/e4m3")
    pr, pc = sf_padded_dims(n, k)
    if scale.numel() != pr * pc:
        raise ValueError(
            f"{prefix}: block-scale numel {scale.numel()} != {pr}*{pc}={pr * pc} (128x4 swizzled)"
        )
    if not scale.is_contiguous():
        raise ValueError(f"{prefix}: block-scale is not contiguous")


def _validate_global_scale(prefix: str, gs: torch.Tensor) -> None:
    if gs.dtype != torch.float32:
        raise ValueError(f"{prefix}: global scale dtype {gs.dtype} != float32")
    if gs.numel() != 1:
        raise ValueError(f"{prefix}: global scale numel {gs.numel()} != 1")
    if gs.is_meta:
        return  # skeleton buffers are validated again after the real load
    if not torch.isfinite(gs).all():
        raise ValueError(f"{prefix}: global scale is not finite")
