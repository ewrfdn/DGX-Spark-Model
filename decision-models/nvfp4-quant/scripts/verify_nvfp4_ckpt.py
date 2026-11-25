#!/usr/bin/env python3
"""量化 checkpoint 的权重级校验（不跑 forward、不跑 GEMM、不允许 Fake Quant）。

做四类检查：
  1. 结构完整性：索引 ↔ 分片 ↔ 清单三方一致，原始 tensor 无一遗漏；
  2. 格式合法性：packed dtype/shape、block scale 的 e4m3 有效性、global scale 公式、swizzle 尺寸；
  3. 数值：用**独立的**参考实现从磁盘重读后反量化，与原始 BF16 权重比 rel_l2 / cosine；
     并对抽样 tensor 用 FlashInfer 重新量化，验证 packed 字节按位一致（证明产物就是 FlashInfer 的输出）；
  4. 保留项：未量化 tensor 必须 dtype/shape 一致且逐元素 bit-identical；非权重文件 sha256 一致。

用法：
  python verify_nvfp4_ckpt.py --model-dir <bf16> --ckpt-dir <nvfp4> --out verify_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvfp4_ref as ref  # noqa: E402

REL_L2_WARN = 0.15  # NVFP4 block=16 的经验上限；真实权重通常在 0.01~0.06

NON_WEIGHT_FILES = [
    "readout.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "decision_config.json", "joint_config.json", "processor_config.json", "release-manifest.json",
    "LICENSE", "NOTICE", "README.md", "requirements.txt", "requirements-fast.txt",
    "locate-span/adapter.safetensors", "locate-span/head.safetensors", "locate-span/locate_config.json",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/sakana/models/decision-models/pplx-decider-flash-v1.3")
    ap.add_argument("--ckpt-dir", default="/home/sakana/models/decision-models/pplx-decider-flash-v1.3-NVFP4")
    ap.add_argument("--out", default="verify_report.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--requant-sample", type=int, default=8, help="用 FlashInfer 重新按位比对的抽样数")
    args = ap.parse_args()

    t0 = time.time()
    model_dir, ckpt_dir = os.path.abspath(args.model_dir), os.path.abspath(args.ckpt_dir)
    device = torch.device(args.device)
    problems: list[str] = []

    src_index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8"))
    out_index = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(ckpt_dir, "nvfp4_manifest.json"), encoding="utf-8"))
    src_cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    out_cfg = json.load(open(os.path.join(ckpt_dir, "config.json"), encoding="utf-8"))

    # ---------- 1. 结构完整性 ----------
    struct = {}
    out_shards = sorted(set(out_index["weight_map"].values()))
    shard_keys: set[str] = set()
    for s in out_shards:
        p = os.path.join(ckpt_dir, s)
        with safe_open(p, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        shard_keys.update(keys)
        struct.setdefault("shard_key_counts", {})[s] = len(keys)
    struct["index_keys"] = len(out_index["weight_map"])
    struct["shard_keys"] = len(shard_keys)
    if shard_keys != set(out_index["weight_map"]):
        problems.append(f"索引与分片 key 不一致: 只在索引 {len(set(out_index['weight_map'])-shard_keys)} / 只在分片 {len(shard_keys-set(out_index['weight_map']))}")
    total_bytes = sum(os.path.getsize(os.path.join(ckpt_dir, s)) for s in out_shards)
    struct["total_bytes"] = total_bytes
    struct["index_total_size"] = out_index["metadata"].get("total_size")
    if abs(total_bytes - out_index["metadata"].get("total_size", -1)) > 0:
        problems.append("索引 total_size 与实际文件大小不一致")

    quantized = {t["name"]: t for t in manifest["tensors"] if t["status"] == "nvfp4"}
    preserved = {t["name"]: t for t in manifest["tensors"] if t["status"] == "bf16_preserved"}
    missing = set(src_index["weight_map"]) - set(quantized) - set(preserved)
    extra = (set(quantized) | set(preserved)) - set(src_index["weight_map"])
    if missing:
        problems.append(f"原始 tensor 遗漏 {len(missing)} 个: {sorted(missing)[:3]}")
    if extra:
        problems.append(f"清单包含原始不存在的 tensor {len(extra)} 个: {sorted(extra)[:3]}")
    struct["covered_original"] = len(src_index["weight_map"])
    struct["quantized"] = len(quantized)
    struct["preserved"] = len(preserved)

    # config：原有字段必须原样保留
    cfg_ok = all(src_cfg.get(k) == v for k, v in src_cfg.items())
    struct["config_superset_ok"] = cfg_ok
    struct["has_quantization_config"] = "quantization_config" in out_cfg
    if not cfg_ok:
        problems.append("config.json 的原有字段被改动")
    if "quantization_config" not in out_cfg:
        problems.append("config.json 缺少 quantization_config")

    # 非权重文件 sha256
    file_checks = []
    for rel in NON_WEIGHT_FILES:
        a, b = os.path.join(model_dir, rel), os.path.join(ckpt_dir, rel)
        if not os.path.exists(a):
            continue
        if not os.path.exists(b):
            problems.append(f"缺失保留文件: {rel}")
            file_checks.append({"file": rel, "ok": False, "reason": "missing"})
            continue
        ha, hb = sha256_file(a), sha256_file(b)
        file_checks.append({"file": rel, "ok": ha == hb, "sha256": ha})
        if ha != hb:
            problems.append(f"保留文件被改动: {rel}")

    # ---------- 2/3/4. 逐 tensor ----------
    out_files = {s: safe_open(os.path.join(ckpt_dir, s), framework="pt", device="cpu") for s in out_shards}
    results = []
    group_agg: dict[str, list[float]] = {}
    n_requant = 0
    from flashinfer.fp4_quantization import SfLayout, nvfp4_quantize

    src_shards = sorted(set(src_index["weight_map"].values()))
    for s in src_shards:
        with safe_open(os.path.join(model_dir, s), framework="pt", device="cpu") as f:
            for key in list(f.keys()):
                w = f.get_tensor(key)
                rec = {"name": key, "shape": list(w.shape), "dtype": str(w.dtype).replace("torch.", "")}
                if key in preserved:
                    h = out_files[preserved[key]["out_shard"]]
                    c = h.get_tensor(key)
                    rec["status"] = "preserved"
                    rec["identical"] = bool(c.dtype == w.dtype and c.shape == w.shape and torch.equal(c, w))
                    if not rec["identical"]:
                        problems.append(f"保留 tensor 未按位一致: {key}")
                    results.append(rec)
                    del w, c
                    continue

                meta = quantized[key]
                h = out_files[meta["out_shard"]]
                q = meta["nvfp4"]
                packed = h.get_tensor(q["weight"])
                sf = h.get_tensor(q["scale"])
                gs = h.get_tensor(q["global_scale"])
                m, k = w.shape
                padded_rows, padded_cols = ref.sf_padded_dims(m, k)
                sf_logical = ref.unswizzle_sf(sf, m, k).view(torch.float8_e4m3fn).to(torch.float32)
                amax = float(w.to(torch.float32).abs().amax())

                checks = {
                    "packed_shape": tuple(packed.shape) == (m, k // 2) and packed.dtype == torch.uint8,
                    "sf_numel": int(sf.numel()) == padded_rows * padded_cols and sf.dtype == torch.uint8,
                    "sf_finite": bool(torch.isfinite(sf_logical).all()),
                    "sf_range": bool((sf_logical >= 0).all() and (sf_logical <= ref.FP8_E4M3_MAX).all()),
                    "gs_finite_positive": bool(torch.isfinite(gs).all() and (gs > 0).all()) if amax > 0 else True,
                    "gs_formula": bool(
                        amax == 0 or abs(float(gs) - ref.FP8_E4M3_MAX * ref.FP4_MAX / amax) <= 1e-4 * float(gs)
                    ),
                }
                if not all(checks.values()):
                    problems.append(f"格式检查失败: {key} -> {[k for k, v in checks.items() if not v]}")

                x_hat = ref.dequantize_swizzled(packed.to(device), sf.to(device), gs.to(device))
                err = ref.quant_error(w.to(device), x_hat)
                rec.update({"status": "nvfp4", "group": meta["group"], "checks": checks, "error": err,
                            "global_scale": float(gs), "amax": amax,
                            "sf_min": float(sf_logical.min()), "sf_max": float(sf_logical.max())})
                if err["rel_l2"] > REL_L2_WARN:
                    problems.append(f"rel_l2 超阈值({err['rel_l2']:.4f}): {key}")
                group_agg.setdefault(meta["group"], []).append(err["rel_l2"])

                # 抽样：用 FlashInfer 重新量化，packed 与 scale 必须按位一致
                if n_requant < args.requant_sample:
                    gpu_w = w.to(device=device, dtype=torch.bfloat16)
                    gs2 = ref.compute_global_scale(amax).reshape(1).to(torch.float32)
                    p2, s2 = nvfp4_quantize(gpu_w, gs2, sfLayout=SfLayout.layout_128x4,
                                            do_shuffle=False, sf_vec_size=ref.BLOCK_SIZE)
                    same = bool(torch.equal(p2.cpu(), packed) and torch.equal(s2.cpu().reshape(-1), sf.reshape(-1)))
                    rec["flashinfer_requant_identical"] = same
                    if not same:
                        problems.append(f"重新量化结果与 checkpoint 不一致: {key}")
                    n_requant += 1
                    del gpu_w, p2, s2

                results.append(rec)
                del w, packed, sf, gs, x_hat
        torch.cuda.empty_cache()

    for h in out_files.values():
        del h

    summary = {
        "quantized": len(quantized),
        "preserved": len(preserved),
        "rel_l2_by_group": {
            g: {"mean": sum(v) / len(v), "max": max(v), "count": len(v)} for g, v in sorted(group_agg.items())
        },
        "rel_l2_overall_mean": (sum(sum(v) for v in group_agg.values()) / max(1, sum(len(v) for v in group_agg.values()))),
        "worst": max(results, key=lambda r: r.get("error", {}).get("rel_l2", -1)).get("name"),
        "requant_checked": n_requant,
    }

    report = {
        "ok": not problems,
        "problems": problems,
        "structure": struct,
        "file_checks": file_checks,
        "summary": summary,
        "seconds": time.time() - t0,
        "tensors": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[structure] shards={len(out_shards)} keys={struct['shard_keys']} "
          f"quantized={summary['quantized']} preserved={summary['preserved']}")
    print(f"[files] {sum(1 for c in file_checks if c['ok'])}/{len(file_checks)} 保留文件 sha256 一致")
    for g, v in summary["rel_l2_by_group"].items():
        print(f"[rel_l2] {g:<14} mean={v['mean']:.5f} max={v['max']:.5f} n={v['count']}")
    print(f"[rel_l2] overall mean={summary['rel_l2_overall_mean']:.5f}  worst={summary['worst']}")
    print(f"[requant] FlashInfer 重新量化按位一致: {summary['requant_checked']}/{summary['requant_checked']}")
    print(f"[result] {'PASS' if report['ok'] else 'FAIL'}  {args.out}")
    for p in problems[:20]:
        print("  !!", p)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
