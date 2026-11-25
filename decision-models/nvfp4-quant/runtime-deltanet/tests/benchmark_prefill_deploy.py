"""Validate the deployable prefill bundle against the baseline on one Decider."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys

import torch

from benchmark_deltanet import (
    MAX_MAE,
    MIN_TOP1,
    Comparison,
    load_requests,
    timed_decision,
    timing_request,
)


@contextmanager
def prefill_candidate(model, install):
    restore = install(model.backbone)
    try:
        yield
    finally:
        restore()


def decision_output(model, response):
    if model.last_logits is None:
        raise ValueError("The request produced no decision logits")
    return response, model.last_logits


def decide_with_logits(model, payload):
    model.last_logits = None
    return decision_output(model, model.decide(payload))


def measure(model, payload):
    model.last_logits = None
    torch.cuda.reset_peak_memory_stats(model.device)
    elapsed_ms, response = timed_decision(model, payload)
    return {
        "latency_ms": elapsed_ms,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
    }, decision_output(model, response)


def regression(model, install, requests):
    baseline_repeat = Comparison()
    comparison = Comparison()
    for index, (request_id, payload) in enumerate(requests):
        if index % 2 == 0:
            baseline = decide_with_logits(model, payload)
        with prefill_candidate(model, install):
            candidate = decide_with_logits(model, payload)
        if index % 2:
            baseline = decide_with_logits(model, payload)
        repeated = decide_with_logits(model, payload)
        baseline_repeat.update(request_id, baseline, repeated)
        comparison.update(request_id, baseline, candidate)
        if (index + 1) % 20 == 0:
            print(f"Evaluated {index + 1}/{len(requests)} regression requests", flush=True)
    return {"baseline_repeat": baseline_repeat.summary(), "candidate": comparison.summary()}


def compare_timing(model, install, token_targets, warmup, repeats):
    results = []
    for question_count in (1, 3):
        for target in token_targets:
            payload = timing_request(model.processor.tokenizer, target, question_count)
            comparison = Comparison()
            samples = {"baseline": [], "candidate": []}
            actual_token_counts = set()
            for index in range(warmup + repeats):
                if index % 2 == 0:
                    baseline, baseline_output = measure(model, payload)
                with prefill_candidate(model, install):
                    candidate, candidate_output = measure(model, payload)
                if index % 2:
                    baseline, baseline_output = measure(model, payload)
                if index >= warmup:
                    samples["baseline"].append(baseline)
                    samples["candidate"].append(candidate)
                    actual_token_counts.add(baseline_output[0]["usage"]["input_tokens"])
                    actual_token_counts.add(candidate_output[0]["usage"]["input_tokens"])
                    comparison.update(index - warmup, baseline_output, candidate_output)
            if len(actual_token_counts) != 1:
                raise ValueError(f"Inconsistent measured input tokens: {actual_token_counts}")
            (tokens,) = actual_token_counts
            baseline_ms = median(record["latency_ms"] for record in samples["baseline"])
            candidate_ms = median(record["latency_ms"] for record in samples["candidate"])
            row = {
                "target_state_tokens": target,
                "input_tokens": tokens,
                "questions_per_request": question_count,
                "baseline_p50_ms": baseline_ms,
                "candidate_p50_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "baseline_tokens_per_second": tokens * 1_000 / baseline_ms,
                "candidate_tokens_per_second": tokens * 1_000 / candidate_ms,
                "samples": samples,
                "comparison": comparison.summary(),
            }
            results.append(row)
            print(f"{tokens} tokens / {question_count} q: {baseline_ms:.2f} → {candidate_ms:.2f} ms", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Existing decisions request JSONL")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2_000, 8_000])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Read only the first N requests; 0 means all")
    args = parser.parse_args()
    if min(args.repeat, *args.tokens) < 1 or min(args.warmup, args.limit) < 0:
        parser.error("Use positive repeat and token counts, and nonnegative warmup and limit")

    if (
        os.environ.get("DECIDER_PREFILL_OPT", "0") != "0"
        or os.environ.get("DECIDER_DELTANET_OPT", "off") != "off"
        or os.environ.get("DECIDER_NVFP4_SHARED_QUANT", "0") != "0"
        or os.environ.get("DECIDER_GDN_BACKEND", "baseline") != "baseline"
    ):
        parser.error("Load the A/B baseline with the prefill bundle, DeltaNet override and shared quant disabled")

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider
    from nvfp4.prefill_optimizations import install_prefill_optimizations

    class BenchmarkDecider(NVFP4Decider):
        def _respond(self, request, logits, chunks, scale):
            self.last_logits = logits
            return super()._respond(request, logits, chunks, scale)

    requests = load_requests(args.dataset, args.limit)
    print(f"Loading {args.checkpoint} for bundle A/B", flush=True)
    model = BenchmarkDecider.from_pretrained(str(args.checkpoint), device="cuda", mode="isolated", fast=False)
    report = {
        "mode": "isolated",
        "baseline": "prefill optimization off",
        "candidate": "DECIDER_PREFILL_OPT=1 runtime bundle",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "runtime": model.runtime_report(),
        "accuracy_thresholds": {"min_top1_agreement": MIN_TOP1, "max_mean_abs_probability_error": MAX_MAE},
        "timing_config": {"warmup_pairs": args.warmup, "measured_pairs": args.repeat, "question_counts": [1, 3]},
    }
    timing = compare_timing(model, install_prefill_optimizations, args.tokens, args.warmup, args.repeat)
    accuracy = regression(model, install_prefill_optimizations, requests)
    report["timing"] = timing
    report["accuracy"] = accuracy
    report["passed"] = all(row["comparison"]["passed"] for row in timing) and all(
        row["passed"] for row in accuracy.values()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
