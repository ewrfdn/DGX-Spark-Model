"""Benchmark and attribute CUDA time for the current W4A4 prefill paths."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmark_deltanet import benchmark, optimized, timing_request


SCOPE = "prefill::"
LAYER_SCOPE = "layer:"
DECODER_LAYERS = {"delta_decoder", "full_decoder"}


def tagged(name, function):
    @wraps(function)
    def call(*args, **kwargs):
        with record_function(SCOPE + name):
            return function(*args, **kwargs)

    return call


@contextmanager
def add_scopes(model):
    from nvfp4 import backend, linear
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    with ExitStack() as stack:
        def wrap(owner, method, scope):
            stack.enter_context(patch.object(owner, method, tagged(scope, getattr(owner, method))))

        wrap(model.backbone, "forward", "backbone")
        wrap(qwen, "torch_chunk_gated_delta_rule", "gdn_core")
        wrap(qwen, "causal_conv1d_fn", "causal_conv")
        wrap(backend, "quantize_activation", "fp4_activation_quant")
        wrap(linear, "mm_fp4", "fp4_gemm")
        for module in model.backbone.language_model.modules():
            if isinstance(module, qwen.Qwen3_5RMSNormGated):
                wrap(module, "forward", "gated_rmsnorm")
            elif isinstance(module, qwen.Qwen3_5RMSNorm):
                wrap(module, "forward", "plain_rmsnorm")
        for index, layer in enumerate(model.backbone.language_model.layers):
            wrap(layer.mlp, "forward", "mlp_other")
            if getattr(layer, "linear_attn", None) is not None:
                wrap(layer, "forward", "delta_decoder")
                wrap(layer.linear_attn, "forward", "delta_layout_and_gates")
                for project in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                    wrap(getattr(layer.linear_attn, project), "forward", "delta_projection_other")
            else:
                wrap(layer, "forward", "full_decoder")
                wrap(layer.self_attn, "forward", "full_attention_other")
            wrap(layer, "forward", f"{LAYER_SCOPE}{index}")
        yield


def event_operators(names):
    operators = [name for name in names if name.startswith("aten::")]
    for name in ("aten::repeat_interleave", "aten::contiguous", "aten::_to_copy"):
        if name in operators:
            return name
    return operators[0] if operators else "direct kernel launch"


def ranked(totals, counts, limit):
    return [{"name": name, "cuda_ms": value / 1000, "calls": counts[name]} for name, value in totals.most_common(limit)]


def correlate_kernels(captured):
    with TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        captured.export_chrome_trace(str(trace))
        with trace.open(encoding="utf-8") as stream:
            events = json.load(stream)["traceEvents"]

    intervals, queries = defaultdict(list), defaultdict(list)
    external_cpu, launch_events = {}, {}
    gpu_events = []
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        args = event.get("args", {})
        correlation = args.get("correlation")
        external = args.get("External id")
        if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            gpu_events.append(event)
        elif category in {"cuda_runtime", "cuda_driver"} and correlation:
            launch_events[correlation] = event
        elif category in {"cpu_op", "user_annotation"}:
            intervals[event["pid"], event["tid"]].append(event)
            if external:
                external_cpu[external] = event

    for gpu_event in gpu_events:
        args = gpu_event.get("args", {})
        launch = launch_events.get(args.get("correlation"))
        if launch is None:
            launch = external_cpu.get(args.get("External id"))
        if launch is None:
            yield gpu_event, None
        else:
            queries[launch["pid"], launch["tid"]].append((launch["ts"], gpu_event))

    for thread, launches in queries.items():
        spans = sorted(intervals[thread], key=lambda event: (event["ts"], -event["dur"]))
        active = []
        start = 0
        for timestamp, gpu_event in sorted(launches, key=lambda row: row[0]):
            while start < len(spans) and spans[start]["ts"] <= timestamp:
                active.append(spans[start])
                start += 1
            active = [event for event in active if timestamp <= event["ts"] + event["dur"]]
            yield gpu_event, [event["name"] for event in reversed(active)]


def summarize(captured):
    stages, layer_totals, all_kernel_us, all_kernel_calls = Counter(), Counter(), Counter(), Counter()
    ops, op_calls, kernels, kernel_calls = (defaultdict(Counter) for _ in range(4))
    layers = defaultdict(Counter)
    raw_us = 0.0
    for event, names in correlate_kernels(captured):
        duration = event["dur"]
        kernel = event["name"]
        raw_us += duration
        all_kernel_us[kernel] += duration
        all_kernel_calls[kernel] += 1
        if names is None:
            continue
        scopes = [name.removeprefix(SCOPE) for name in names if name.startswith(SCOPE)]
        layer = next((name for name in scopes if name.startswith(LAYER_SCOPE)), None)
        if layer is not None:
            scopes.remove(layer)
        stage = next((name for name in scopes if name not in DECODER_LAYERS), "outside_backbone")
        if "aten::scaled_dot_product_attention" in names:
            stage = "sdpa"
        operation = event_operators(names)
        stages[stage] += duration
        if layer is not None:
            layers[int(layer.removeprefix(LAYER_SCOPE))][stage] += duration
        for scope in DECODER_LAYERS.intersection(scopes):
            layer_totals[scope] += duration
        ops[stage][operation] += duration
        op_calls[stage][operation] += 1
        kernels[stage][kernel] += duration
        kernel_calls[stage][kernel] += 1
    attributed = stages.total()
    if not raw_us or not attributed:
        raise RuntimeError("CUPTI did not return CUDA kernel attribution")
    return {
        "raw_cuda_ms": raw_us / 1000,
        "attributed_cuda_ms": attributed / 1000,
        "attributed_fraction": attributed / raw_us,
        "decoder_cuda_ms_including_nested_stages": {name: value / 1000 for name, value in layer_totals.items()},
        "decoder_layers": {
            index: {name: value / 1000 for name, value in layer_stages.items()}
            for index, layer_stages in sorted(layers.items())
        },
        "exclusive_stages": {
            name: {
                "cuda_ms": value / 1000,
                "fraction_of_attributed": value / attributed,
                "top_operators": ranked(ops[name], op_calls[name], 8),
                "top_kernels": ranked(kernels[name], kernel_calls[name], 8),
            }
            for name, value in stages.most_common()
        },
        "top_kernels_all_stages": ranked(all_kernel_us, all_kernel_calls, 12),
    }


def trace_request(model, payload, candidate):
    with optimized(model.backbone, "both") if candidate else nullcontext():
        model.decide(payload)
        torch.cuda.synchronize()
        with add_scopes(model), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            response = model.decide(payload)
            torch.cuda.synchronize()
    return {"input_tokens": response["usage"]["input_tokens"], **summarize(captured)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tokens", nargs="+", type=int, default=[2000, 8000])
    parser.add_argument("--questions", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--timings-from", type=Path, help="Reuse already measured wall timings in a separate profiler pass")
    args = parser.parse_args()
    if min(*args.tokens, *args.questions, args.repeat) < 1 or args.warmup < 0:
        parser.error("Lengths, question counts and repeats must be positive")

    sys.path.insert(0, str(Path(args.checkpoint) / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    model = NVFP4Decider.from_pretrained(args.checkpoint, device="cuda", mode="isolated", fast=False)
    if model.gdn_backend != "baseline" or getattr(model, "deltanet_opt", "off") != "off":
        raise ValueError("Use the baseline GDN backend with the DeltaNet switches off for paired runs")
    report = {
        "runtime": model.runtime_report(),
        "warmup_pairs": args.warmup,
        "measured_pairs": args.repeat,
        "timing_notes": "Wall timings exclude profiler. CUDA times sum kernel and transfer activity, not wall time.",
        "runs": [],
    }
    timings = {}
    if args.timings_from:
        measured = json.loads(args.timings_from.read_text(encoding="utf-8"))
        report.update({key: measured[key] for key in ("warmup_pairs", "measured_pairs")})
        timings = {
            (row["questions"], row["timing"]["target_state_tokens"]): row["timing"] for row in measured["runs"]
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for questions in args.questions:
        for target in args.tokens:
            if args.timings_from:
                timing = timings[questions, target]
            else:
                timing, = benchmark(model, "both", [target], args.warmup, args.repeat, questions)
            workload = timing_request(model.processor.tokenizer, target, questions)
            row = {"questions": questions, "timing": timing}
            for label in ("baseline", "optimized"):
                row[label] = trace_request(model, workload, label == "optimized")
                if row[label]["input_tokens"] != timing["input_tokens"]:
                    raise RuntimeError("Profiling and wall timing workloads have different input token counts")
            report["runs"].append(row)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Profiled {timing['input_tokens']} input tokens and {questions} questions", flush=True)


if __name__ == "__main__":
    main()
