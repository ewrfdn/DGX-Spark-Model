"""Same-process A/B timing, 200-request regression, and 32-layer profiling for P0+P1."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
from statistics import median
import sys
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark_deltanet import (
    MAX_MAE,
    MIN_TOP1,
    Comparison,
    load_requests,
    timed_decision,
    timing_request,
)
from prefill_v2 import PrefillV2Config, install_prefill_v2
from profile_current_prefill import summarize, tagged


VARIANTS: dict[str, PrefillV2Config] = {
    "p1_only": PrefillV2Config(
        shared_quant=True,
        inline_amax=True,
        fused_mlp_swiglu=True,
        fused_residual_rmsnorm=True,
        fused_qk_rope_and_gate=True,
        gva_native=False,
        split_isolated_deltanet=False,
        flash_full_attention=False,
        prune_last_layer=False,
    ),
    "p1_plus_gva": PrefillV2Config(
        shared_quant=True,
        inline_amax=True,
        fused_mlp_swiglu=True,
        fused_residual_rmsnorm=True,
        fused_qk_rope_and_gate=True,
        gva_native=True,
        split_isolated_deltanet=False,
        flash_full_attention=False,
        prune_last_layer=False,
    ),
    "p0_p1_no_prune": PrefillV2Config(
        shared_quant=True,
        inline_amax=True,
        fused_mlp_swiglu=True,
        fused_residual_rmsnorm=True,
        fused_qk_rope_and_gate=True,
        gva_native=True,
        split_isolated_deltanet=True,
        flash_full_attention=True,
        prune_last_layer=False,
    ),
    "p0_p1_full": PrefillV2Config(
        shared_quant=True,
        inline_amax=True,
        fused_mlp_swiglu=True,
        fused_residual_rmsnorm=True,
        fused_qk_rope_and_gate=True,
        gva_native=True,
        split_isolated_deltanet=True,
        flash_full_attention=True,
        prune_last_layer=True,
    ),
}


@contextmanager
def installed_variant(model, cfg: PrefillV2Config, *, profile_scopes: bool = False):
    restore = install_prefill_v2(model, cfg, profile_scopes=profile_scopes)
    try:
        yield
    finally:
        restore()


def decide_with_logits(model, payload):
    model.last_logits = None
    resp = model.decide(payload)
    if model.last_logits is None:
        raise ValueError("No logits captured")
    return resp, model.last_logits


def measure_one(model, payload):
    model.last_logits = None
    torch.cuda.reset_peak_memory_stats(model.device)
    ms, resp = timed_decision(model, payload)
    peak = torch.cuda.max_memory_allocated(model.device)
    return {"latency_ms": ms, "peak_torch_allocated_bytes": peak}, (resp, model.last_logits)


def run_timing(model, variant_name: str, cfg: PrefillV2Config, token_targets, questions_list, warmup: int, repeat: int):
    rows = []
    for q in questions_list:
        for target in token_targets:
            payload = timing_request(model.processor.tokenizer, target, q)
            samples = {"baseline": [], "candidate": []}
            comp = Comparison()
            toks = set()
            for idx in range(warmup + repeat):
                if idx % 2 == 0:
                    b_stat, b_out = measure_one(model, payload)
                with installed_variant(model, cfg):
                    c_stat, c_out = measure_one(model, payload)
                if idx % 2 == 1:
                    b_stat, b_out = measure_one(model, payload)
                if idx >= warmup:
                    samples["baseline"].append(b_stat)
                    samples["candidate"].append(c_stat)
                    toks.add(b_out[0]["usage"]["input_tokens"])
                    toks.add(c_out[0]["usage"]["input_tokens"])
                    comp.update(idx - warmup, b_out, c_out)
            (tokens,) = toks
            b_ms = median(r["latency_ms"] for r in samples["baseline"])
            c_ms = median(r["latency_ms"] for r in samples["candidate"])
            b_mem = max(r["peak_torch_allocated_bytes"] for r in samples["baseline"])
            c_mem = max(r["peak_torch_allocated_bytes"] for r in samples["candidate"])
            row = {
                "variant": variant_name,
                "questions": q,
                "target_state_tokens": target,
                "input_tokens": tokens,
                "baseline_p50_ms": b_ms,
                "candidate_p50_ms": c_ms,
                "speedup": b_ms / c_ms,
                "baseline_tok_s": tokens * 1000.0 / b_ms,
                "candidate_tok_s": tokens * 1000.0 / c_ms,
                "baseline_peak_gb": b_mem / (1024**3),
                "candidate_peak_gb": c_mem / (1024**3),
                "comparison": comp.summary(),
                "samples": samples,
            }
            rows.append(row)
            s = row["comparison"]
            print(
                f"[{variant_name}] Q={q} tok={tokens}: {b_ms:.2f} -> {c_ms:.2f} ms "
                f"({row['speedup']:.2f}x, {row['candidate_tok_s']:.0f} tok/s) | "
                f"top1={s['top1_agreement']*100:.2f}% mae={s['mean_abs_probability_error']:.6f} "
                f"bit_eq={s['bitwise_equal_logit_questions']}/{s['questions']} "
                f"peak={row['candidate_peak_gb']:.2f}GB",
                flush=True,
            )
    return rows


def run_regression(model, variant_name: str, cfg: PrefillV2Config, requests, modes):
    out = {}
    for mode in modes:
        model.mode = mode
        aa, ab = Comparison(), Comparison()
        for idx, (req_id, payload) in enumerate(requests):
            if idx % 2 == 0:
                base = decide_with_logits(model, payload)
            with installed_variant(model, cfg):
                cand = decide_with_logits(model, payload)
            if idx % 2 == 1:
                base = decide_with_logits(model, payload)
            rep = decide_with_logits(model, payload)
            aa.update(req_id, base, rep)
            ab.update(req_id, base, cand)
            if (idx + 1) % 40 == 0 or (idx + 1) == len(requests):
                print(f"[{variant_name}][{mode}] evaluated {idx + 1}/{len(requests)} requests", flush=True)
        out[mode] = {"baseline_repeat": aa.summary(), "candidate": ab.summary()}
        c = out[mode]["candidate"]
        print(
            f"[{variant_name}][{mode}] SUMMARY: passed={c['passed']} top1={c['top1_agreement']*100:.2f}% "
            f"mae={c['mean_abs_probability_error']:.6f} max_prob_err={c['max_abs_probability_error']:.6f} "
            f"max_logit_err={c['max_abs_logit_error']:.6f} bit_eq={c['bitwise_equal_logit_questions']}/{c['questions']}",
            flush=True,
        )
    return out


def run_layer_profile(model, variant_name: str, cfg: PrefillV2Config, token_targets, questions_list, warmup: int, repeat: int):
    from nvfp4 import backend as nv_backend
    from nvfp4 import linear as nv_linear

    model.mode = "isolated"
    layer_types = [layer.block_type for layer in model.backbone.language_model.layers]
    runs = []
    with installed_variant(model, cfg, profile_scopes=True):
        with ExitStack() as stack:
            stack.enter_context(patch.object(nv_backend, "quantize_activation", tagged("fp4_activation_quant", nv_backend.quantize_activation)))
            stack.enter_context(patch.object(nv_linear, "mm_fp4", tagged("fp4_gemm", nv_linear.mm_fp4)))
            for q in questions_list:
                for target in token_targets:
                    payload = timing_request(model.processor.tokenizer, target, q)
                    for _ in range(warmup):
                        model.decide(payload)
                    torch.cuda.synchronize(model.device)

                    samples = []
                    for _ in range(repeat):
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as cap:
                            resp = model.decide(payload)
                            torch.cuda.synchronize(model.device)
                        res = summarize(cap)
                        dec_total = sum(res["decoder_cuda_ms_including_nested_stages"].values())
                        samples.append({
                            "input_tokens": resp["usage"]["input_tokens"],
                            "raw_cuda_ms": res["raw_cuda_ms"],
                            "decoder_cuda_ms": dec_total,
                            "layers": res["decoder_layers"],
                            "exclusive_stages": res["exclusive_stages"],
                            "top_kernels_all_stages": res["top_kernels_all_stages"],
                        })
                    layers_out = []
                    for idx, kind in enumerate(layer_types):
                        ls = [s["layers"].get(idx, {}) for s in samples]
                        times = [sum(st.values()) for st in ls]
                        labels = sorted({k for st in ls for k in st})
                        layers_out.append({
                            "index": idx,
                            "type": kind,
                            "cuda_p50_ms": median(times),
                            "stage_p50_ms": {k: median(st.get(k, 0.0) for st in ls) for k in labels},
                        })
                    raw_p50 = median(s["raw_cuda_ms"] for s in samples)
                    dec_p50 = median(s["decoder_cuda_ms"] for s in samples)
                    runs.append({
                        "variant": variant_name,
                        "questions": q,
                        "target_state_tokens": target,
                        "input_tokens": samples[0]["input_tokens"],
                        "raw_cuda_p50_ms": raw_p50,
                        "decoder_cuda_p50_ms": dec_p50,
                        "layers": layers_out,
                        "last_sample_exclusive_stages": samples[-1]["exclusive_stages"],
                        "last_sample_top_kernels": samples[-1]["top_kernels_all_stages"],
                    })
                    print(
                        f"[profile][{variant_name}] Q={q} tok={samples[0]['input_tokens']}: "
                        f"raw_cuda_p50={raw_p50:.2f}ms decoder_p50={dec_p50:.2f}ms",
                        flush=True,
                    )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bf16-checkpoint", type=Path, help="Optional BF16 checkpoint for ground-truth accuracy comparison")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=["p1_only", "p1_plus_gva", "p0_p1_no_prune", "p0_p1_full"], choices=tuple(VARIANTS))
    parser.add_argument("--regression-variants", nargs="+", default=["p1_only", "p0_p1_full"], choices=tuple(VARIANTS))
    parser.add_argument("--profile-variant", default="p0_p1_full", choices=tuple(VARIANTS))
    parser.add_argument("--tokens", nargs="+", type=int, default=[2000, 8000])
    parser.add_argument("--questions", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    class BenchmarkDecider(NVFP4Decider):
        def _respond(self, request, logits, chunks, scale):
            self.last_logits = logits
            return super()._respond(request, logits, chunks, scale)

    bf16_ref = {}
    if not args.skip_regression and args.dataset is not None and args.bf16_checkpoint is not None:
        from pplx_decider import Decider

        class BF16Decider(Decider):
            def _respond(self, request, logits, chunks, scale):
                self.last_logits = logits
                return super()._respond(request, logits, chunks, scale)

        requests = load_requests(args.dataset, args.limit)
        print(f"Loading BF16 reference {args.bf16_checkpoint} for {len(requests)}-request ground truth...", flush=True)
        bf16_model = BF16Decider.from_pretrained(str(args.bf16_checkpoint), device="cuda", mode="isolated", fast=False)
        for mode in ("isolated", "separate"):
            bf16_model.mode = mode
            bf16_ref[mode] = [decide_with_logits(bf16_model, payload) for _, payload in requests]
            print(f"[bf16][{mode}] collected {len(requests)} reference outputs", flush=True)
        del bf16_model
        torch.cuda.empty_cache()

    print(f"Loading {args.checkpoint}...", flush=True)
    model = BenchmarkDecider.from_pretrained(str(args.checkpoint), device="cuda", mode="isolated", fast=False)

    report: dict = {
        "runtime": model.runtime_report(),
        "accuracy_thresholds": {"min_top1_agreement": MIN_TOP1, "max_mean_abs_probability_error": MAX_MAE},
        "timing": {},
        "regression": {},
        "vs_bf16": {},
        "profile": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_timing:
        model.mode = "isolated"
        for name in args.variants:
            report["timing"][name] = run_timing(
                model, name, VARIANTS[name], args.tokens, args.questions, args.warmup, args.repeat
            )
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_regression and args.dataset is not None:
        requests = load_requests(args.dataset, args.limit)
        # First collect baseline outputs once per mode (for both vs-baseline and vs-bf16)
        base_outputs = {}
        for mode in ("isolated", "separate"):
            model.mode = mode
            base_outputs[mode] = [decide_with_logits(model, payload) for _, payload in requests]
            if mode in bf16_ref:
                c_bf16 = Comparison()
                for (req_id, _), ref_o, b_o in zip(requests, bf16_ref[mode], base_outputs[mode]):
                    c_bf16.update(req_id, ref_o, b_o)
                s = c_bf16.summary()
                report["vs_bf16"].setdefault("baseline", {})[mode] = s
                print(
                    f"[baseline vs BF16][{mode}] top1={s['top1_agreement']*100:.2f}% "
                    f"mae={s['mean_abs_probability_error']:.6f} max_p={s['max_abs_probability_error']:.6f}",
                    flush=True,
                )
        for name in args.regression_variants:
            report["regression"][name] = {}
            for mode in ("isolated", "separate"):
                model.mode = mode
                with installed_variant(model, VARIANTS[name]):
                    cand_outputs = [decide_with_logits(model, payload) for _, payload in requests]
                c_base = Comparison()
                for (req_id, _), b_o, c_o in zip(requests, base_outputs[mode], cand_outputs):
                    c_base.update(req_id, b_o, c_o)
                s_base = c_base.summary()
                report["regression"][name][mode] = s_base
                print(
                    f"[{name} vs NVFP4 base][{mode}] passed={s_base['passed']} top1={s_base['top1_agreement']*100:.2f}% "
                    f"mae={s_base['mean_abs_probability_error']:.6f} max_p={s_base['max_abs_probability_error']:.6f} "
                    f"bit_eq={s_base['bitwise_equal_logit_questions']}/{s_base['questions']}",
                    flush=True,
                )
                if mode in bf16_ref:
                    c_bf16 = Comparison()
                    for (req_id, _), ref_o, c_o in zip(requests, bf16_ref[mode], cand_outputs):
                        c_bf16.update(req_id, ref_o, c_o)
                    s_bf = c_bf16.summary()
                    report["vs_bf16"].setdefault(name, {})[mode] = s_bf
                    print(
                        f"[{name} vs BF16][{mode}] top1={s_bf['top1_agreement']*100:.2f}% "
                        f"mae={s_bf['mean_abs_probability_error']:.6f} max_p={s_bf['max_abs_probability_error']:.6f}",
                        flush=True,
                    )
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_profile:
        report["profile"] = run_layer_profile(
            model, args.profile_variant, VARIANTS[args.profile_variant], args.tokens, args.questions, warmup=1, repeat=3
        )
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
