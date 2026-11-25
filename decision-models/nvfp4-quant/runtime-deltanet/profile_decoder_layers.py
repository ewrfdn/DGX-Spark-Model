"""Measure each decoder layer in the optimized Decider prefill path."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from statistics import median
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark_deltanet import optimized, timing_request
from profile_current_prefill import add_scopes, summarize


def measure(model, request):
    with add_scopes(model), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
        response = model.decide(request)
        torch.cuda.synchronize(model.device)
    result = summarize(captured)
    layer_total = sum(sum(stages.values()) for stages in result["decoder_layers"].values())
    decoder_total = sum(result["decoder_cuda_ms_including_nested_stages"].values())
    if not math.isclose(layer_total, decoder_total, abs_tol=0.0001):
        raise RuntimeError(f"Decoder and individual layer CUDA times differ: {decoder_total}, {layer_total}")
    return {
        "input_tokens": response["usage"]["input_tokens"],
        "raw_cuda_ms": result["raw_cuda_ms"],
        "decoder_cuda_ms": decoder_total,
        "layers": result["decoder_layers"],
    }


def group_samples(layer_types, samples):
    input_tokens = {sample["input_tokens"] for sample in samples}
    if len(input_tokens) != 1:
        raise RuntimeError(f"Input token count changed across samples: {input_tokens}")
    if any(set(sample["layers"]) != set(range(len(layer_types))) for sample in samples):
        raise RuntimeError("A CUDA trace missed one or more decoder layers")

    layers = []
    for index, kind in enumerate(layer_types):
        layer_samples = [sample["layers"][index] for sample in samples]
        times = [sum(stages.values()) for stages in layer_samples]
        labels = sorted({label for stages in layer_samples for label in stages})
        layers.append({
            "index": index,
            "type": kind,
            "cuda_p50_ms": median(times),
            "cuda_samples_ms": times,
            "stage_p50_ms": {label: median(stages.get(label, 0.0) for stages in layer_samples) for label in labels},
        })
    return {
        "input_tokens": input_tokens.pop(),
        "raw_cuda_p50_ms": median(sample["raw_cuda_ms"] for sample in samples),
        "decoder_cuda_p50_ms": median(sample["decoder_cuda_ms"] for sample in samples),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tokens", nargs="+", type=int, default=[2000, 8000])
    parser.add_argument("--questions", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--plain-rmsnorm", action="store_true", help="Also test exact-order Triton ordinary RMSNorm")
    args = parser.parse_args()
    if min(*args.tokens, *args.questions, args.warmup, args.repeat) < 1:
        parser.error("Lengths, question counts, warmup and repeats must be positive")

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    model = NVFP4Decider.from_pretrained(str(args.checkpoint), device="cuda", mode="isolated", fast=False)
    if model.gdn_backend != "baseline" or getattr(model, "deltanet_opt", "off") != "off":
        raise ValueError("Use the baseline GDN backend and enable DeltaNet through this script")
    layer_types = [layer.block_type for layer in model.backbone.language_model.layers]
    report = {
        "runtime": model.runtime_report(),
        "deltanet_optimization": "both",
        "plain_rmsnorm": "triton_reduction" if args.plain_rmsnorm else "baseline",
        "warmup_requests": args.warmup,
        "profiled_requests": args.repeat,
        "timing_notes": "CUPTI CUDA work attributed by launch scope. Layer index is zero-based.",
        "runs": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(optimized(model.backbone, "both"))
        if args.plain_rmsnorm:
            from plain_rmsnorm import install_plain_rmsnorm

            stack.callback(install_plain_rmsnorm(model.backbone, triton_reduction=True))
        for questions in args.questions:
            for target in args.tokens:
                request = timing_request(model.processor.tokenizer, target, questions)
                for _ in range(args.warmup):
                    model.decide(request)
                torch.cuda.synchronize(model.device)
                samples = [measure(model, request) for _ in range(args.repeat)]
                row = group_samples(layer_types, samples)
                report["runs"].append({"questions": questions, "target_state_tokens": target, **row})
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"Profiled {row['input_tokens']} input tokens, {questions} questions and {len(layer_types)} layers", flush=True)


if __name__ == "__main__":
    main()
