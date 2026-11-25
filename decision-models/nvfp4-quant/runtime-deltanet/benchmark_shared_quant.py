"""Measure activation quantization reuse with both DeltaNet optimizations enabled."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
from statistics import median
import sys

import torch

from benchmark_deltanet import Comparison, load_requests, optimized, timed_decision, timing_request


@contextmanager
def shared_quant(enabled):
    key = "DECIDER_NVFP4_SHARED_QUANT"
    previous = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def cached_forward(original, quant_cache_scope):
    @wraps(original)
    def forward(*args, **kwargs):
        with quant_cache_scope():
            return original(*args, **kwargs)

    return forward


@contextmanager
def layer_quant_cache(model):
    from nvfp4.backend import quant_cache_scope

    # Each decoder layer contains every consumer of its cached inputs. Nested
    # scopes also cover all Gated DeltaNet calls made within isolated restarts.
    with ExitStack() as stack:
        for layer in model.backbone.language_model.layers:
            had_override = "forward" in vars(layer)
            original = layer.forward
            layer.forward = cached_forward(original, quant_cache_scope)
            if had_override:
                stack.callback(setattr, layer, "forward", original)
            else:
                stack.callback(delattr, layer, "forward")
        yield


@contextmanager
def run_variant(model, variant, scope):
    enabled = variant == "shared"
    cache = layer_quant_cache(model) if enabled and scope == "layer" else nullcontext()
    with shared_quant(enabled), cache:
        yield


def variant_order(index):
    return ("baseline", "shared") if index % 2 == 0 else ("shared", "baseline")


def compare_timing(model, targets, warmup, repeat, scope):
    rows = []
    for questions in (1, 3):
        for target in targets:
            payload = timing_request(model.processor.tokenizer, target, questions)
            samples = {"baseline": [], "shared": []}
            comparison = Comparison()
            repeated_baseline = Comparison()
            earlier_baseline = None
            actual_tokens = set()
            for index in range(warmup + repeat):
                outputs = {}
                for variant in variant_order(index):
                    with run_variant(model, variant, scope):
                        torch.cuda.reset_peak_memory_stats(model.device)
                        model.last_logits = None
                        elapsed_ms, response = timed_decision(model, payload)
                        peak_bytes = torch.cuda.max_memory_allocated(model.device)
                    if model.last_logits is None:
                        raise ValueError(f"No logits captured for {variant}")
                    if index >= warmup:
                        outputs[variant] = (response, model.last_logits)
                        actual_tokens.add(response["usage"]["input_tokens"])
                        samples[variant].append({"latency_ms": elapsed_ms, "peak_torch_allocated_bytes": peak_bytes})
                if index >= warmup:
                    comparison.update(index - warmup, outputs["baseline"], outputs["shared"])
                    if earlier_baseline is not None:
                        repeated_baseline.update(index - warmup, earlier_baseline, outputs["baseline"])
                    earlier_baseline = outputs["baseline"]

            if len(actual_tokens) != 1:
                raise ValueError(f"Inconsistent measured input token counts: {actual_tokens}")
            (tokens,) = actual_tokens
            baseline_ms = median(row["latency_ms"] for row in samples["baseline"])
            shared_ms = median(row["latency_ms"] for row in samples["shared"])
            result = {
                "target_state_tokens": target,
                "input_tokens": tokens,
                "questions_per_request": questions,
                "baseline_p50_ms": baseline_ms,
                "shared_p50_ms": shared_ms,
                "speedup": baseline_ms / shared_ms,
                "baseline_tokens_per_second": tokens * 1_000 / baseline_ms,
                "shared_tokens_per_second": tokens * 1_000 / shared_ms,
                "samples": samples,
                "comparison": comparison.summary(),
                "baseline_repeat": repeated_baseline.summary() if repeat > 1 else None,
            }
            rows.append(result)
            print(f"{tokens} tokens / {questions} q: {baseline_ms:.2f} → {shared_ms:.2f} ms", flush=True)
    return rows


def compare_dataset(model, requests, scope):
    comparison = Comparison()
    for index, (request_id, payload) in enumerate(requests):
        outputs = {}
        for variant in variant_order(index):
            with run_variant(model, variant, scope):
                model.last_logits = None
                response = model.decide(payload)
            if model.last_logits is None:
                raise ValueError(f"No logits captured for {request_id} ({variant})")
            outputs[variant] = (response, model.last_logits)
        comparison.update(request_id, outputs["baseline"], outputs["shared"])
    return comparison.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--scope", choices=("request", "layer"), default="request")
    parser.add_argument("--tokens", type=int, nargs="+", default=[2_000, 8_000])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if min(args.repeat, *args.tokens) < 1 or min(args.warmup, args.limit) < 0:
        parser.error("Use positive repeat and token counts, and nonnegative warmup and limit")

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    class BenchmarkDecider(NVFP4Decider):
        def _respond(self, request, logits, chunks, scale):
            self.last_logits = logits
            return super()._respond(request, logits, chunks, scale)

    model = BenchmarkDecider.from_pretrained(str(args.checkpoint), device="cuda", mode="isolated", fast=False)
    if model.gdn_backend != "baseline" or getattr(model, "deltanet_opt", "off") != "off":
        raise ValueError("Use DECIDER_GDN_BACKEND=baseline and DECIDER_DELTANET_OPT=off")
    report = {
        "mode": "isolated",
        "deltanet_optimization": "both",
        "shared_quant_scope": args.scope,
        "runtime": model.runtime_report(),
        "warmup_pairs": args.warmup,
        "measured_pairs": args.repeat,
        "dataset": str(args.dataset) if args.dataset else None,
    }
    with optimized(model.backbone, "both"):
        timing = compare_timing(model, args.tokens, args.warmup, args.repeat, args.scope)
        report["timing"] = timing
        if args.dataset:
            report["regression"] = compare_dataset(model, load_requests(args.dataset, args.limit), args.scope)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}", flush=True)
    passed = all(
        row["comparison"]["passed"] and (
            row["baseline_repeat"] is None or row["baseline_repeat"]["passed"]
        )
        for row in timing
    )
    return 0 if passed and report.get("regression", {"passed": True})["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
