"""Opt-in fused BF16 gated RMSNorm for the Decider's SM121 runtime."""

from collections.abc import Callable
from functools import wraps

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gated_rms_norm_kernel(
    hidden_ptr, gate_ptr, weight_ptr, output_ptr, rows, eps: tl.constexpr, block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, 128)
    offset = row[:, None] * 128 + column[None, :]
    valid = row[:, None] < rows
    hidden = tl.load(hidden_ptr + offset, valid, other=0).to(tl.float32)
    square = hidden * hidden

    # Match ATen's float4 mean: sum each lane's four consecutive elements
    # in order, then reduce those 32 partial sums within one warp.
    even, odd = tl.split(tl.reshape(square, (block_rows, 32, 2, 2)))
    first, third = tl.split(even)
    second, fourth = tl.split(odd)
    total = tl.sum(((first + second) + third) + fourth, axis=1)
    inv_rms = tl.rsqrt(total * (1.0 / 128) + eps)

    # Transformers rounds after normalization and again after BF16 weighting.
    normed = (hidden * inv_rms[:, None]).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(weight_ptr + column).to(tl.float32)
    weighted = (normed * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(gate_ptr + offset, valid, other=0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    tl.store(output_ptr + offset, weighted * silu, valid)


def _can_fuse(hidden: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and hidden.is_cuda
        and hidden.dtype == gate.dtype == weight.dtype == torch.bfloat16
        and hidden.device == gate.device == weight.device
        and hidden.ndim == 2
        and hidden.shape == gate.shape
        and weight.shape == (128,)
        and hidden.shape[1] == 128
        and hidden.shape[0] >= 16
        and hidden.is_contiguous()
        and gate.is_contiguous()
        and weight.is_contiguous()
    )


def _fused_gated_norm(hidden: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(hidden)
    rows = hidden.shape[0]
    block_rows = 4
    with torch.cuda.device(hidden.device):
        _gated_rms_norm_kernel[(triton.cdiv(rows, block_rows),)](
            hidden, gate, weight, output, rows, eps, block_rows, num_warps=4, enable_fp_fusion=False,
        )
    return output


def _wrap_forward(module: torch.nn.Module) -> Callable:
    forward = module.forward

    @wraps(forward)
    def fused_forward(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if _can_fuse(hidden_states, gate, module.weight):
            return _fused_gated_norm(hidden_states, gate, module.weight, module.variance_epsilon)
        return forward(hidden_states, gate)

    return fused_forward


def install_deltanet_norm(model: torch.nn.Module) -> Callable[[], None]:
    """Install on this model instance and return a function restoring the original forwards."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    modules = [module for module in model.modules() if type(module) is Qwen3_5RMSNormGated]
    if not modules:
        raise ValueError("No Qwen3.5 gated RMSNorm modules found")
    patches = []
    for module in modules:
        original = vars(module).get("forward")
        replacement = _wrap_forward(module)
        module.forward = replacement
        patches.append((module, original, replacement))

    def restore() -> None:
        for module, forward, replacement in patches:
            if vars(module).get("forward") is not replacement:
                continue
            if forward is None:
                del module.forward
            else:
                module.forward = forward

    return restore
