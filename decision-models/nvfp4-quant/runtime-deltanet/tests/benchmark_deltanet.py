"""Compare DeltaNet optimizations against the same W4A4 model in one process."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


MODES = ("separate", "isolated")
MIN_TOP1 = 0.995
MAX_MAE = 0.005


@contextmanager
def optimized(backbone, optimization):
    with ExitStack() as stack:
        if optimization in ("norm", "both"):
            from nvfp4.deltanet_norm import install_deltanet_norm

            stack.callback(install_deltanet_norm(backbone))
        if optimization in ("conv", "both"):
            from nvfp4.deltanet_conv import install_deltanet_conv

            stack.callback(install_deltanet_conv(backbone))
        yield


def probabilities(answer):
    if answer["type"] == "noul":
        p_true = float(answer["noul"])
        return {"false": 1.0 - p_true, "true": p_true}
    return answer["probabilities"]


def top1(answer, distribution):
    if answer["type"] == "noul":
        return distribution["true"] >= 0.5
    if answer["type"] == "choice":
        return answer["choice"]
    return max(distribution, key=distribution.__getitem__)


class Comparison:
    def __init__(self):
        self.requests = 0
        self.usage_mismatch = 0
        self.structure_mismatch = 0
        self.questions = []

    def update(self, request_id, reference, other):
        ref_response, ref_logits = reference
        other_response, other_logits = other
        self.requests += 1
        self.usage_mismatch += ref_response["usage"] != other_response["usage"]
        ref_answers = ref_response["answers"]
        other_answers = other_response["answers"]
        if not (ref_answers.keys() == other_answers.keys() == ref_logits.keys() == other_logits.keys()):
            self.structure_mismatch += 1
            return
        for key, ref_answer in ref_answers.items():
            other_answer = other_answers[key]
            if ref_answer.keys() != other_answer.keys() or ref_answer["type"] != other_answer["type"]:
                self.structure_mismatch += 1
                continue
            ref_probs = probabilities(ref_answer)
            other_probs = probabilities(other_answer)
            if ref_probs.keys() != other_probs.keys() or len(ref_logits[key]) != len(other_logits[key]):
                self.structure_mismatch += 1
                continue
            errors = [abs(ref_probs[option] - other_probs[option]) for option in ref_probs]
            logits = list(zip(ref_logits[key], other_logits[key], strict=True))
            self.questions.append(
                {
                    "id": request_id,
                    "question": key,
                    "type": ref_answer["type"],
                    "top1_agrees": top1(ref_answer, ref_probs) == top1(other_answer, other_probs),
                    "mean_abs_probability_error": statistics.fmean(errors),
                    "max_abs_probability_error": max(errors),
                    "logits_bitwise_equal": all(left.hex() == right.hex() for left, right in logits),
                    "max_abs_logit_error": max(abs(left - right) for left, right in logits),
                }
            )

    def summary(self):
        if not self.questions:
            raise ValueError("No comparable decision questions were evaluated")
        rows = self.questions
        top1_agreement = statistics.fmean(row["top1_agrees"] for row in rows)
        mae = statistics.fmean(row["mean_abs_probability_error"] for row in rows)
        return {
            "passed": (
                not self.structure_mismatch
                and not self.usage_mismatch
                and top1_agreement >= MIN_TOP1
                and mae <= MAX_MAE
            ),
            "requests": self.requests,
            "questions": len(rows),
            "top1_agreement": top1_agreement,
            "mean_abs_probability_error": mae,
            "max_abs_probability_error": max(row["max_abs_probability_error"] for row in rows),
            "bitwise_equal_logit_questions": sum(row["logits_bitwise_equal"] for row in rows),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in rows),
            "usage_mismatch": self.usage_mismatch,
            "structure_mismatch": self.structure_mismatch,
            "worst_questions": sorted(rows, key=lambda row: row["max_abs_probability_error"], reverse=True)[:5],
        }


def evaluate(model, requests, optimization):
    aa, ab = Comparison(), Comparison()
    for index, (request_id, payload) in enumerate(requests):
        if index % 2 == 0:
            baseline = model.decide_with_logits(payload)
        with optimized(model.backbone, optimization):
            candidate = model.decide_with_logits(payload)
        if index % 2:
            baseline = model.decide_with_logits(payload)
        repeated = model.decide_with_logits(payload)
        aa.update(request_id, baseline, repeated)
        ab.update(request_id, baseline, candidate)
        if (index + 1) % 20 == 0:
            print(f"[{model.mode}] evaluated {index + 1}/{len(requests)} requests", flush=True)
    return {"baseline_repeat": aa.summary(), "candidate": ab.summary()}


def timing_request(tokenizer, tokens, question_count=1):
    sentence = (
        "The customer reports an intermittent 401 after key rotation and duplicated billing records. "
        "They attached gateway logs and asked for the next step. "
    )
    state = sentence * (tokens // len(tokenizer.encode(sentence, add_special_tokens=False)))
    while len(tokenizer.encode(state, add_special_tokens=False)) < tokens:
        state += sentence
    questions = {"escalate": {"type": "noul", "instructions": "Does this need a human?"}}
    for index in range(1, question_count):
        questions[f"followup_{index}"] = {
            "type": "noul", "instructions": f"Does the customer ask for a follow-up action numbered {index}?",
        }
    return {"state": state, "questions": questions}


def timed_decision(model, payload):
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    response = model.decide(payload)
    torch.cuda.synchronize(model.device)
    return (time.perf_counter() - started) * 1_000, response


def benchmark(model, optimization, token_targets, warmup, repeat, question_count=1):
    rows = []
    for target in token_targets:
        payload = timing_request(model.processor.tokenizer, target, question_count)
        latencies = {"baseline": [], "candidate": []}
        input_tokens = set()
        for index in range(warmup + repeat):
            variants = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for variant in variants:
                if variant == "candidate":
                    with optimized(model.backbone, optimization):
                        elapsed_ms, response = timed_decision(model, payload)
                else:
                    elapsed_ms, response = timed_decision(model, payload)
                if index >= warmup:
                    input_tokens.add(response["usage"]["input_tokens"])
                    latencies[variant].append(elapsed_ms)
        if len(input_tokens) != 1:
            raise RuntimeError(f"Input token count changed during timing: {input_tokens}")
        (tokens,) = input_tokens
        baseline_ms = statistics.median(latencies["baseline"])
        candidate_ms = statistics.median(latencies["candidate"])
        row = {
            "target_state_tokens": target,
            "input_tokens": tokens,
            "baseline_p50_ms": baseline_ms,
            "candidate_p50_ms": candidate_ms,
            "speedup": baseline_ms / candidate_ms,
            "baseline_tokens_per_second": tokens * 1_000 / baseline_ms,
            "candidate_tokens_per_second": tokens * 1_000 / candidate_ms,
            "latencies_ms": latencies,
        }
        rows.append(row)
        print(f"[{model.mode}] {tokens} tokens, {baseline_ms:.2f} → {candidate_ms:.2f} ms", flush=True)
    return rows


def load_requests(dataset, limit):
    with dataset.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    requests = [
        (row.get("id", index), {key: value for key, value in row.items() if key not in {"id", "long"}})
        for index, row in enumerate(rows[:limit] if limit else rows)
    ]
    if not requests:
        raise ValueError(f"Empty evaluation set: {dataset}")
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, help="Existing decisions request JSONL")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--optimization", required=True, choices=("norm", "conv", "both"))
    parser.add_argument("--mode", choices=(*MODES, "both"), default="both")
    parser.add_argument("--tokens", type=int, nargs="+", default=[2_000, 8_000])
    parser.add_argument("--isolated-questions", type=int, default=1, help="Timing workload only")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Read only the first N requests; 0 means all")
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--timing-only", action="store_true")
    steps.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    if not args.timing_only and args.dataset is None:
        parser.error("--dataset is required for decision evaluation")
    if min(args.repeat, args.isolated_questions, *args.tokens) < 1 or min(args.warmup, args.limit) < 0:
        parser.error("Timings require positive repeat, tokens and question counts, and nonnegative warmup and limit")

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    class BenchmarkDecider(NVFP4Decider):
        def _respond(self, request, logits, chunks, scale):
            self.last_logits = logits
            return super()._respond(request, logits, chunks, scale)

        def decide_with_logits(self, payload):
            self.last_logits = None
            response = self.decide(payload)
            if self.last_logits is None:
                raise ValueError("The evaluation request did not produce decision logits")
            return response, self.last_logits

    modes = MODES if args.mode == "both" else (args.mode,)
    print(f"Loading {args.checkpoint} for {args.optimization} A/B", flush=True)
    model = BenchmarkDecider.from_pretrained(str(args.checkpoint), device="cuda", mode=modes[0], fast=False)
    if model.gdn_backend != "baseline":
        raise ValueError("Run with DECIDER_GDN_BACKEND=baseline to measure the reference DeltaNet path")
    if getattr(model, "deltanet_opt", "off") != "off":
        raise ValueError("Run with DECIDER_DELTANET_OPT=off so the A/B baseline is eager")
    requests = [] if args.timing_only else load_requests(args.dataset, args.limit)
    report = {
        "optimization": args.optimization,
        "dataset": str(args.dataset) if args.dataset else None,
        "accuracy_thresholds": {"min_top1_agreement": MIN_TOP1, "max_mean_abs_probability_error": MAX_MAE},
        "timing_config": {
            "warmup_pairs": args.warmup, "measured_pairs": args.repeat, "isolated_questions": args.isolated_questions,
        },
        "runtime": model.runtime_report(),
        "modes": {},
    }
    passed = True
    for mode in modes:
        model.mode = mode
        results = {}
        if not args.skip_timing:
            questions = args.isolated_questions if mode == "isolated" else 1
            results["timing"] = benchmark(model, args.optimization, args.tokens, args.warmup, args.repeat, questions)
        if requests:
            accuracy = evaluate(model, requests, args.optimization)
            results["accuracy"] = accuracy
            passed &= all(comparison["passed"] for comparison in accuracy.values())
        report["modes"][mode] = results
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
