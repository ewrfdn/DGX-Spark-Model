"""A/B ordinary RMSNorm with both verified DeltaNet kernels active."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

import benchmark_deltanet as benchmark
from nvfp4.plain_rmsnorm import install_plain_rmsnorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--variant", choices=("epilogue", "triton_reduction"), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2_000, 8_000])
    parser.add_argument("--single-question-mode", choices=("isolated", "separate"), default="isolated")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if min(*args.tokens, args.warmup, args.repeat, args.limit) < 1:
        parser.error("--tokens, --warmup, --repeat and --limit must be positive")

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    class ProbeDecider(NVFP4Decider):
        def _respond(self, request, logits, chunks, scale):
            self.last_logits = logits
            return super()._respond(request, logits, chunks, scale)

        def decide_with_logits(self, payload):
            self.last_logits = None
            response = self.decide(payload)
            if self.last_logits is None:
                raise ValueError("The evaluation request did not produce decision logits")
            return response, self.last_logits

    @contextmanager
    def plain_only(backbone, variant):
        restore = install_plain_rmsnorm(backbone, triton_reduction=variant == "triton_reduction")
        try:
            yield
        finally:
            restore()

    model = ProbeDecider.from_pretrained(str(args.checkpoint), device="cuda", mode=args.single_question_mode, fast=False)
    if model.gdn_backend != "baseline" or getattr(model, "deltanet_opt", "off") != "off":
        raise ValueError("Run with DECIDER_GDN_BACKEND=baseline and DECIDER_DELTANET_OPT=off")
    requests = benchmark.load_requests(args.dataset, args.limit)
    report = {
        "variant": args.variant,
        "baseline": "W4A4 with fused DeltaNet gated RMSNorm and causal convolution",
        "runtime": model.runtime_report(),
        "dataset": str(args.dataset),
        "dataset_requests_per_mode": len(requests),
        "warmup_pairs": args.warmup,
        "measured_pairs": args.repeat,
        "modes": {},
    }
    exact = True
    with benchmark.optimized(model.backbone, "both"), patch.object(benchmark, "optimized", plain_only):
        for mode, questions in ((args.single_question_mode, 1), ("isolated", 3)):
            model.mode = mode
            timing = benchmark.benchmark(model, args.variant, args.tokens, args.warmup, args.repeat, questions)
            latency_requests = [
                (f"timing_{target}", benchmark.timing_request(model.processor.tokenizer, target, questions))
                for target in args.tokens
            ]
            accuracy = benchmark.evaluate(model, requests + latency_requests, args.variant)
            for comparison in accuracy.values():
                exact &= (
                    comparison["passed"]
                    and comparison["bitwise_equal_logit_questions"] == comparison["questions"]
                )
            report["modes"][f"{mode}_q{questions}"] = {
                "timing_questions": questions, "timing": timing, "accuracy": accuracy,
            }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}", flush=True)
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
