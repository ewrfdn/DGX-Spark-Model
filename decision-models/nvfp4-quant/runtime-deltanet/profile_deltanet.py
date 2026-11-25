"""Attribute W4A4 DeltaNet time to submodules using paired CUDA events."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, nullcontext
from functools import wraps
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import torch

from benchmark_deltanet import MODES, optimized, timing_request


class Spans:
    def __init__(self):
        self.events = defaultdict(list)
        self.shapes = defaultdict(Counter)
        self.dispatches = Counter()

    def timed(self, stage, function, token_axis=None):
        @wraps(function)
        def call(*args, **kwargs):
            if token_axis is not None:
                shape = args[0].shape
                self.shapes[stage][shape[0], shape[token_axis]] += 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = function(*args, **kwargs)
            end.record()
            self.events[stage].append((start, end))
            return result

        return call

    def count(self, kernel, function):
        @wraps(function)
        def call(*args, **kwargs):
            self.dispatches[kernel] += 1
            return function(*args, **kwargs)

        return call

    def result(self):
        stages = {
            name: {"ms": sum(start.elapsed_time(end) for start, end in pairs), "calls": len(pairs)}
            for name, pairs in self.events.items()
        }
        return {
            "stages": stages,
            "linear_attn_other_ms": stages["linear_attn"]["ms"] - sum(
                stages[name]["ms"] for name in ("causal_conv1d", "gdn_core", "gated_rms_norm")
            ),
            "shapes": {
                name: [{"batch": batch, "tokens": tokens, "calls": calls} for (batch, tokens), calls in sorted(counts.items())]
                for name, counts in self.shapes.items()
            },
            "custom_dispatches": {name: self.dispatches[name] for name in ("gated_rms_norm", "causal_conv1d")},
        }


def profile_request(model, payload, candidate):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    spans = Spans()
    with torch.cuda.device(model.device), ExitStack() as stack:
        if candidate:
            import deltanet_conv
            import deltanet_norm

            stack.enter_context(patch.object(
                deltanet_conv, "causal_conv4_silu", spans.count("causal_conv1d", deltanet_conv.causal_conv4_silu),
            ))
            stack.enter_context(patch.object(
                deltanet_norm, "_fused_gated_norm", spans.count("gated_rms_norm", deltanet_norm._fused_gated_norm),
            ))
        stack.enter_context(patch.object(qwen, "causal_conv1d_fn", spans.timed("causal_conv1d", qwen.causal_conv1d_fn, 2)))
        stack.enter_context(patch.object(
            qwen, "torch_chunk_gated_delta_rule", spans.timed("gdn_core", qwen.torch_chunk_gated_delta_rule, 1),
        ))
        for layer in model.backbone.language_model.layers:
            if not isinstance(getattr(layer, "linear_attn", None), qwen.Qwen3_5GatedDeltaNet):
                continue
            for stage, module in (
                ("linear_attn", layer.linear_attn),
                ("gated_rms_norm", layer.linear_attn.norm),
                ("input_layernorm", layer.input_layernorm),
                ("post_attention_layernorm", layer.post_attention_layernorm),
            ):
                stack.enter_context(patch.object(module, "forward", spans.timed(stage, module.forward)))
        torch.cuda.synchronize(model.device)
        response = spans.timed("decision", model.decide)(payload)
        torch.cuda.synchronize(model.device)
    result = spans.result()
    result["input_tokens"] = response["usage"]["input_tokens"]
    return result


def summarize(samples):
    stages = samples[0]["stages"]
    return {
        "stages": {
            stage: {
                "p50_ms": statistics.median(row["stages"][stage]["ms"] for row in samples),
                "calls_per_request": [row["stages"][stage]["calls"] for row in samples],
            }
            for stage in stages
        },
        "linear_attn_other_p50_ms": statistics.median(row["linear_attn_other_ms"] for row in samples),
        "samples": samples,
    }


def compare(model, target, question_count, warmup, repeat):
    payload = timing_request(model.processor.tokenizer, target, question_count)

    samples = {"baseline": [], "both": []}
    for index in range(warmup + repeat):
        variants = ("baseline", "both") if index % 2 == 0 else ("both", "baseline")
        for variant in variants:
            candidate = variant == "both"
            with optimized(model.backbone, "both") if candidate else nullcontext():
                if index < warmup:
                    model.decide(payload)
                    torch.cuda.synchronize(model.device)
                else:
                    samples[variant].append(profile_request(model, payload, candidate))

    input_tokens = {row["input_tokens"] for variant in samples.values() for row in variant}
    if len(input_tokens) != 1:
        raise RuntimeError(f"Usage differed across variants: {input_tokens}")
    (actual_tokens,) = input_tokens
    result = {"target_state_tokens": target, "input_tokens": actual_tokens, "questions": question_count}
    result.update({variant: summarize(rows) for variant, rows in samples.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--mode", choices=(*MODES, "both"), default="both")
    parser.add_argument("--tokens", type=int, nargs="+", default=[2_000])
    parser.add_argument(
        "--isolated-questions", type=int, default=3, help="Three questions trigger a batch of two restarted sequences",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if min(*args.tokens, args.isolated_questions, args.warmup, args.repeat) < 1:
        parser.error("--tokens, --isolated-questions, --warmup and --repeat must be positive")

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    modes = MODES if args.mode == "both" else (args.mode,)
    model = NVFP4Decider.from_pretrained(str(args.checkpoint), device="cuda", mode=modes[0], fast=False)
    if model.gdn_backend != "baseline":
        raise ValueError("Run with DECIDER_GDN_BACKEND=baseline to profile the FLA core")
    report = {
        "runtime": model.runtime_report(),
        "warmup_pairs": args.warmup,
        "measured_pairs": args.repeat,
        "timing_notes": (
            "CUDA event intervals bracket Python dispatch. Nested stages overlap linear_attn. "
            "input_layernorm and post_attention_layernorm cover only DeltaNet decoder layers. "
            "Use the uninstrumented benchmark for end-to-end throughput."
        ),
        "modes": {},
    }
    for mode in modes:
        model.mode = mode
        questions = args.isolated_questions if mode == "isolated" else 1
        rows = []
        for target in args.tokens:
            rows.append(compare(model, target, questions, args.warmup, args.repeat))
            print(f"[{mode}] profiled {rows[-1]['input_tokens']} tokens and {questions} questions", flush=True)
        report["modes"][mode] = rows

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
