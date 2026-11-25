"""Experimental exact BF16 ordinary RMSNorm epilogue for the Decider."""

from collections.abc import Callable
from functools import wraps

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_rms_kernel(
    x, inverse, rows, row_stride, width: tl.constexpr, epsilon: tl.constexpr, block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, 128)
    start = row[:, None].to(tl.int64) * row_stride + column[None, :]
    valid = row[:, None] < rows
    first = tl.full((block_rows, 32), 0, tl.float32)
    second = tl.full((block_rows, 32), 0, tl.float32)
    third = tl.full((block_rows, 32), 0, tl.float32)
    fourth = tl.full((block_rows, 32), 0, tl.float32)

    # Match the FP32 ATen vectorized mean for these widths and at least 16 rows.
    # Each lane accumulates four independent streams before the warp reduction.
    for wave in tl.range(width // 128, loop_unroll_factor=2):
        values = tl.load(x + start + wave * 128, valid, other=0).to(tl.float32)
        even, odd = tl.split(tl.reshape(values * values, (block_rows, 32, 2, 2)))
        a, c = tl.split(even)
        b, d = tl.split(odd)
        first += a
        second += b
        third += c
        fourth += d

    total = tl.sum(((first + second) + third) + fourth, axis=1)
    tl.store(inverse + row, tl.rsqrt(total * (1.0 / width) + epsilon), row < rows)


@triton.jit
def _rms_epilogue(
    x, weight, inverse, output, rows, row_stride, width: tl.constexpr,
    block_rows: tl.constexpr, block_columns: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    valid = row[:, None] < rows
    x_offset = row[:, None].to(tl.int64) * row_stride + column[None, :]
    output_offset = row[:, None].to(tl.int64) * width + column[None, :]
    values = tl.load(x + x_offset, valid, other=0).to(tl.float32)
    inv = tl.load(inverse + row, row < rows, other=0)
    weights = tl.load(weight + column).to(tl.float32)
    normalized = values * inv[:, None]
    tl.store(output + output_offset, normalized * (1.0 + weights[None, :]), valid)


def _eligible_view(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.device != weight.device
        or x.ndim < 2
        or x.shape[-1] not in (256, 2560)
        or weight.shape != (x.shape[-1],)
        or not weight.is_contiguous()
        or x.stride(-1) != 1
        or torch.is_grad_enabled()
    ):
        return None
    try:
        flat = x.view(-1, x.shape[-1])
    except RuntimeError:
        return None
    if flat.shape[0] < 16 or flat.stride(0) < flat.shape[1]:
        return None
    return flat


def _plain_rmsnorm(
    flat: torch.Tensor, weight: torch.Tensor, epsilon: float, triton_reduction: bool,
) -> torch.Tensor:
    rows, width = flat.shape
    output = torch.empty(flat.shape, dtype=flat.dtype, device=flat.device)
    with torch.cuda.device(flat.device):
        if triton_reduction:
            inverse = torch.empty(rows, dtype=torch.float32, device=flat.device)
            _inverse_rms_kernel[(triton.cdiv(rows, 4),)](
                flat, inverse, rows, flat.stride(0), width, epsilon, 4,
                num_warps=4, enable_fp_fusion=False,
            )
        else:
            inverse = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + epsilon)
        block_rows, block_columns = (4, 256) if width == 256 else (2, 512)
        _rms_epilogue[(triton.cdiv(rows, block_rows), width // block_columns)](
            flat, weight, inverse, output, rows, flat.stride(0), width, block_rows, block_columns,
            num_warps=4, enable_fp_fusion=False,
        )
    return output


def _wrap_forward(module: torch.nn.Module, triton_reduction: bool) -> Callable:
    eager = module.forward

    @wraps(eager)
    def forward(x: torch.Tensor) -> torch.Tensor:
        flat = _eligible_view(x, module.weight)
        if flat is None:
            return eager(x)
        return _plain_rmsnorm(flat, module.weight, module.eps, triton_reduction).view(x.shape)

    return forward


def install_plain_rmsnorm(model: torch.nn.Module, *, triton_reduction: bool = False) -> Callable[[], None]:
    """Install the epilogue on ordinary Qwen3.5 norms and return an undo callable."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    modules = [module for module in model.modules() if type(module) is Qwen3_5RMSNorm]
    if not modules:
        raise ValueError("No ordinary Qwen3.5 RMSNorm modules found")
    originals = [(module, vars(module).get("forward")) for module in modules]
    for module in modules:
        module.forward = _wrap_forward(module, triton_reduction)

    def restore() -> None:
        for module, original in originals:
            if original is None:
                del module.forward
            else:
                module.forward = original

    return restore
