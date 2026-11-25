"""Token-major BF16 causal depthwise convolution for Qwen3.5 prefill."""

import weakref
from collections.abc import Callable
from functools import wraps

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _causal_conv4_silu(
    x,
    weight,
    out,
    length,
    batch_stride,
    channels: tl.constexpr,
    block_tokens: tl.constexpr,
    block_channels: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    channel = tl.program_id(0) * block_channels + tl.arange(0, block_channels)
    token = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    output_mask = (token[:, None] < length) & (channel[None, :] < channels)
    channel_offset = channel[None, :]
    input_offset = batch * batch_stride + channel_offset
    acc = tl.full((block_tokens, block_channels), 0, tl.float32)
    for tap in tl.static_range(4):
        source_token = token[:, None] + tap - 3
        x_tap = tl.load(
            x + input_offset + source_token.to(tl.int64) * channels,
            output_mask & (source_token >= 0),
            other=0,
        ).to(tl.float32)
        w_tap = tl.load(weight + channel * 4 + tap, channel < channels, other=0).to(tl.float32)
        acc = acc + x_tap * w_tap[None, :]

    # Eager computes SiLU on the already-rounded BF16 convolution result.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    activated = tl.div_rn(rounded, 1.0 + libdevice.exp(-rounded))
    out_offset = (batch * length + token[:, None].to(tl.int64)) * channels + channel_offset
    tl.store(out + out_offset, activated.to(tl.bfloat16), output_mask)


def _supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device == weight.device
        and x.ndim == 3
        and x.shape[0] > 0
        and x.shape[1] > 0
        and x.shape[2] > 1
        and x.stride(1) == 1
        and x.stride(2) == x.shape[1]
        and weight.shape == (x.shape[1], 4)
        and weight.is_contiguous()
    )


def causal_conv4_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return causal conv + SiLU in [B, C, T], with token-major storage."""
    if not _supported(x, weight):
        raise ValueError("Expected token-major CUDA BF16 input [B, C, T>1] and contiguous weight [C, 4]")

    batch, channels, length = x.shape
    out = x.new_empty((batch, length, channels))
    block_tokens, block_channels = 8, 128
    with torch.cuda.device(x.device):
        _causal_conv4_silu[
            (triton.cdiv(channels, block_channels), triton.cdiv(length, block_tokens), batch)
        ](
            x,
            weight,
            out,
            length,
            x.stride(0),
            channels,
            block_tokens,
            block_channels,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out.transpose(1, 2)


def install_deltanet_conv(model: torch.nn.Module) -> Callable[[], None]:
    """Route eligible convolutions from this model to Triton and return an undo callable."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    weights = {
        (module.conv1d.weight.device, module.conv1d.weight.data_ptr()): weakref.ref(module.conv1d.weight)
        for module in model.modules()
        if isinstance(module, qwen.Qwen3_5GatedDeltaNet)
        and module.conv1d.weight.is_cuda
        and module.conv1d.bias is None
        and module.conv_kernel_size == 4
    }
    if not weights:
        raise ValueError("The model has no supported Qwen3.5 CUDA convolution weights")

    original = qwen.causal_conv1d_fn

    @wraps(original)
    def dispatch(hidden_states, weight, bias=None, activation=None, **kwargs):
        if (
            not torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
            and bias is None
            and activation == "silu"
            and kwargs.get("seq_idx") is None
            and kwargs.get("cu_seq_lens_q") is None
            and kwargs.get("cu_seqlens") is None
            and _supported(hidden_states, weight)
        ):
            pointer = weight.data_ptr()
            parameter_ref = weights.get((weight.device, pointer))
            parameter = parameter_ref() if parameter_ref is not None else None
            if parameter is not None and parameter.device == weight.device and parameter.data_ptr() == pointer:
                return causal_conv4_silu(hidden_states, weight)
        return original(hidden_states, weight, bias, activation=activation, **kwargs)

    qwen.causal_conv1d_fn = dispatch

    def restore():
        weights.clear()
        if qwen.causal_conv1d_fn is dispatch:
            qwen.causal_conv1d_fn = original

    return restore
