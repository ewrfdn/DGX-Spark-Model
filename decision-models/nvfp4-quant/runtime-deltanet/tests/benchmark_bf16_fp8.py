"""Benchmark and accuracy regression for BF16 & FP8 (tensor/rowwise) + Prefill V2.

Compares against the untouched BF16 eager baseline on:
  * 200-request / 531-question dataset (Top-1 agreement, MAE, Max |dp|, P95)
  * Latency & throughput at 2k / 4k / 8k (Q=1 and Q=3) + 16k / 31.5k (Q=1)
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nvfp4.fp8 import install_fp8_linears
from nvfp4.prefill_v2 import PrefillV2Config, install_prefill_v2


def load_eval_requests(path: Path) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        payload = {"state": row["state"], "questions": row["questions"]}
        if row.get("images"):
            payload["images"] = row["images"]
        out.append((str(row.get("id", i)), payload))
    return out


def extract_probs(ans: dict[str, Any]) -> tuple[str, list[float], Any]:
    qtype = ans.get("type", "")
    if "probabilities" in ans and isinstance(ans["probabilities"], dict):
        keys = sorted(ans["probabilities"].keys())
        probs = [float(ans["probabilities"][k]) for k in keys]
        top1 = ans.get("choice", max(keys, key=lambda k: ans["probabilities"][k]))
        return qtype, probs, top1
    if "noul" in ans:
        p = float(ans["noul"])
        return qtype, [p, 1.0 - p], (p >= 0.5)
    if "score" in ans:
        p = float(ans["score"])
        return qtype, [p], round(p, 2)
    raise ValueError(f"Unexpected answer format: {ans}")


def compare_responses(ref_list: list[dict[str, Any]], cand_list: list[dict[str, Any]]) -> dict[str, Any]:
    top1_ok = 0
    total_q = 0
    maes: list[float] = []
    maxes: list[float] = []
    by_type: dict[str, dict[str, Any]] = {}

    for ref, cand in zip(ref_list, cand_list, strict=True):
        for qk, rans in ref["answers"].items():
            cans = cand["answers"][qk]
            qtype, rprobs, rtop = extract_probs(rans)
            _, cprobs, ctop = extract_probs(cans)
            diffs = [abs(a - b) for a, b in zip(rprobs, cprobs, strict=True)]
            q_mae = sum(diffs) / len(diffs)
            q_max = max(diffs)
            agree = (rtop == ctop)
            top1_ok += int(agree)
            total_q += 1
            maes.append(q_mae)
            maxes.append(q_max)
            st = by_type.setdefault(qtype, {"ok": 0, "n": 0, "max_errs": []})
            st["ok"] += int(agree)
            st["n"] += 1
            st["max_errs"].append(q_max)

    sorted_max = sorted(maxes)
    p95 = sorted_max[int(0.95 * (len(sorted_max) - 1))] if sorted_max else 0.0
    return {
        "requests": len(ref_list),
        "questions": total_q,
        "top1_agreement": top1_ok / total_q if total_q else 1.0,
        "top1_matches": top1_ok,
        "mean_abs_prob_error": sum(maes) / len(maes) if maes else 0.0,
        "p95_abs_prob_error": p95,
        "max_abs_prob_error": max(maxes) if maxes else 0.0,
        "by_type": {
            k: {
                "top1_agreement": v["ok"] / v["n"],
                "n": v["n"],
                "max_abs_prob_error": max(v["max_errs"]),
            }
            for k, v in by_type.items()
        },
    }


def make_timing_req(tokenizer, target_tokens: int, num_questions: int) -> dict[str, Any]:
    para = (
        "The customer reports an intermittent 401 after key rotation and duplicated billing records. "
        "They attached gateway logs and asked for the next step. "
    )
    state = para
    while len(tokenizer.encode(state, add_special_tokens=False)) < target_tokens:
        cur = len(tokenizer.encode(state, add_special_tokens=False))
        state = (state + state) if cur * 2 < target_tokens else (state + para)
    ids = tokenizer.encode(state, add_special_tokens=False)[:target_tokens]
    questions: dict[str, Any] = {"escalate": {"type": "noul", "instructions": "Does this need a human?"}}
    for idx in range(1, num_questions):
        questions[f"followup_{idx}"] = {
            "type": "noul",
            "instructions": f"Does the customer ask for a follow-up action numbered {idx}?",
        }
    return {"state": tokenizer.decode(ids), "questions": questions}


def bench_timing(decider, cases: list[tuple[int, int]], warmup: int = 1, repeat: int = 3) -> list[dict[str, Any]]:
    tok = decider.processor.tokenizer
    rows = []
    for target, nq in cases:
        payload = make_timing_req(tok, target, nq)
        for _ in range(warmup):
            decider.decide(payload)
        torch.cuda.synchronize(decider.device)
        torch.cuda.reset_peak_memory_stats(decider.device)
        lats = []
        in_tok = 0
        for _ in range(repeat):
            t0 = time.perf_counter()
            resp = decider.decide(payload)
            torch.cuda.synchronize(decider.device)
            lats.append((time.perf_counter() - t0) * 1000.0)
            in_tok = resp["usage"]["input_tokens"]
        p50 = statistics.median(lats)
        peak_gb = torch.cuda.max_memory_allocated(decider.device) / 1e9
        row = {
            "target_tokens": target,
            "questions": nq,
            "input_tokens": in_tok,
            "p50_ms": round(p50, 2),
            "tok_s": round(in_tok / (p50 / 1000.0), 1),
            "peak_gb": round(peak_gb, 2),
            "samples_ms": [round(x, 2) for x in lats],
        }
        rows.append(row)
        print(
            f"  Q={nq} target={target:5d} in_tok={in_tok:5d} | p50={p50:7.2f} ms | {row['tok_s']:7.1f} tok/s | peak={peak_gb:.2f} GB",
            flush=True,
        )
    return rows


def run_eval_all(decider, reqs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    outs = []
    t0 = time.perf_counter()
    for i, (_, payload) in enumerate(reqs):
        outs.append(decider.decide(payload))
        if (i + 1) % 50 == 0:
            print(f"  eval {i + 1}/{len(reqs)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    torch.cuda.synchronize(decider.device)
    return outs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mode", default="isolated")
    args = parser.parse_args()

    from pplx_decider import Decider

    reqs = load_eval_requests(Path(args.dataset))
    base_cases = [(2000, 1), (4000, 1), (8000, 1), (2000, 3), (8000, 3)]
    full_cases = [(2000, 1), (4000, 1), (8000, 1), (16000, 1), (31500, 1), (2000, 3), (8000, 3)]

    report: dict[str, Any] = {"checkpoint": args.checkpoint, "mode": args.mode, "variants": {}}

    # 1. BF16 baseline + BF16 prefill_v2=full (same model instance)
    print("=== [1/4] Loading BF16 checkpoint ===", flush=True)
    dec = Decider.from_pretrained(args.checkpoint, device="cuda", mode=args.mode, fast=False)
    print("=== [1/4] BF16 baseline (prefill_v2=off) ===", flush=True)
    bf16_base_timing = bench_timing(dec, base_cases, warmup=1, repeat=3)
    bf16_ref_eval = run_eval_all(dec, reqs)
    report["variants"]["bf16_baseline"] = {"timing": bf16_base_timing}

    print("=== [2/4] BF16 + prefill_v2=full ===", flush=True)
    restore = install_prefill_v2(dec, PrefillV2Config())
    bf16_v2_timing = bench_timing(dec, full_cases, warmup=1, repeat=3)
    bf16_v2_eval = run_eval_all(dec, reqs)
    bf16_v2_acc = compare_responses(bf16_ref_eval, bf16_v2_eval)
    print("  BF16+V2 vs BF16 base:", json.dumps(bf16_v2_acc), flush=True)
    report["variants"]["bf16_prefill_v2"] = {"timing": bf16_v2_timing, "vs_bf16": bf16_v2_acc}
    restore()
    del restore, dec
    gc.collect()
    torch.cuda.empty_cache()

    # 3. FP8 (tensor) + prefill_v2=full
    for idx, scaling in [(3, "tensor"), (4, "rowwise")]:
        print(f"=== [{idx}/4] FP8 ({scaling}) + prefill_v2=full ===", flush=True)
        dec = Decider.from_pretrained(args.checkpoint, device="cuda", mode=args.mode, fast=False)
        fp8_info = install_fp8_linears(dec.backbone, scaling=scaling, use_fast_accum=False)
        print("  FP8 installed:", fp8_info, flush=True)
        restore_fp8 = install_prefill_v2(dec, PrefillV2Config())
        timing = bench_timing(dec, full_cases, warmup=1, repeat=3)
        eval_out = run_eval_all(dec, reqs)
        acc = compare_responses(bf16_ref_eval, eval_out)
        print(f"  FP8({scaling})+V2 vs BF16 base:", json.dumps(acc), flush=True)
        report["variants"][f"fp8_{scaling}_prefill_v2"] = {
            "fp8_info": fp8_info,
            "timing": timing,
            "vs_bf16": acc,
        }
        restore_fp8()
        del restore_fp8, dec
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
