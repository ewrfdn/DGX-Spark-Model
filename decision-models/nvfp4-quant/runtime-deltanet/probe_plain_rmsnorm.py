"""Measure ordinary RMSNorm variants on the Decider's prefill shapes."""

import argparse
import json
from pathlib import Path

import torch

from plain_rmsnorm import _eligible_view, install_plain_rmsnorm


def _elapsed_ms(function, warmup, repeat):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _probe_case(shape, stride_factor, warmup, repeat):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    width = shape[-1]
    storage = torch.randn((*shape[:-1], width * stride_factor), device="cuda", dtype=torch.bfloat16)
    hidden = storage[..., :width]
    module = Qwen3_5RMSNorm(width, eps=1e-6).to(device="cuda", dtype=torch.bfloat16).eval()
    module.weight.normal_(0.0, 0.13)
    if _eligible_view(hidden, module.weight) is None:
        raise RuntimeError(f"Probe input unexpectedly fell back: {shape}, strides={hidden.stride()}")
    reference = module(hidden)
    baseline_ms = _elapsed_ms(lambda: module(hidden), warmup, repeat)
    results = {"shape": shape, "strides": hidden.stride(), "baseline_ms": baseline_ms, "variants": {}}
    for name, triton_reduction in (("epilogue", False), ("triton_reduction", True)):
        restore = install_plain_rmsnorm(module, triton_reduction=triton_reduction)
        try:
            actual = module(hidden)
            changed = torch.count_nonzero(actual.view(torch.int16) != reference.view(torch.int16)).item()
            elapsed_ms = _elapsed_ms(lambda: module(hidden), warmup, repeat)
        finally:
            restore()
        results["variants"][name] = {
            "changed_elements": changed,
            "elements": reference.numel(),
            "max_abs_error": (actual.float() - reference.float()).abs().max().item(),
            "ms": elapsed_ms,
            "speedup": baseline_ms / elapsed_ms,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2_086, 8_078])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.tokens) < 16 or min(args.warmup, args.repeat) < 1:
        parser.error("--tokens must be at least 16, --warmup and --repeat must be positive")

    torch.manual_seed(2026)
    with torch.inference_mode():
        rows = [
            _probe_case(shape, stride_factor, args.warmup, args.repeat)
            for tokens in args.tokens
            for shape, stride_factor in (
                ((1, tokens, 2_560), 1),
                ((1, tokens, 4, 256), 1),
                ((1, tokens, 16, 256), 2),
            )
        ]
    report = json.dumps({"gpu": torch.cuda.get_device_name(), "cases": rows}, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return int(any(variant["changed_elements"] for row in rows for variant in row["variants"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
