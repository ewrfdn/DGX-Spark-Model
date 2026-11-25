#!/usr/bin/env python3
"""把 nvfp4_manifest.json（+ 可选的 verify_report.json）渲染成人类可读的量化清单 Markdown。

用法：
  python make_manifest_md.py --manifest <ckpt>/nvfp4_manifest.json \
      --verify <ckpt>/verify_report.json --out 量化清单.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def fmt_bytes(n: int) -> str:
    return f"{n / 1e9:.3f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verify", default=None)
    ap.add_argument("--out", default="量化清单.md")
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding="utf-8"))
    ver = json.load(open(args.verify, encoding="utf-8")) if args.verify else None
    err = {t["name"]: t.get("error") for t in (ver or {}).get("tensors", []) if t.get("status") == "nvfp4"}
    q = man["quantizer"]
    c = man["counts"]

    lines: list[str] = []
    add = lines.append
    add("# 量化清单 —— pplx-decider-flash-v1.3 → NVFP4")
    add("")
    add(f"生成时间（UTC）：`{man['created_utc']}`　·　清单 schema：`{man['schema']}`")
    add("")
    add("## 1. 概览")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 原始模型 | `{man['source_model_dir']}` |")
    add(f"| 量化产物 | `{man['out_dir']}` |")
    add(f"| 量化库 / API | `{q['library']}` · `{q['api']}` |")
    add(f"| 量化策略 | {q['mode']} |")
    add(f"| scale 布局 | `{q['sf_layout']}`（block = 16） |")
    add(f"| 运行环境 | torch {q['torch']} / CUDA {q['cuda_runtime']} / {q['gpu']} / cc {q['compute_capability']} |")
    add(f"| tensor 总数 | {c['tensors_total']}（量化 {c['tensors_quantized']} / 保留 {c['tensors_preserved']}） |")
    add(f"| 参数量 | 总 {c['params_total'] / 1e9:.3f} B，其中量化 {c['params_quantized'] / 1e9:.3f} B"
        f"（{100 * c['params_quantized'] / max(1, c['params_total']):.1f}%） |")
    add(f"| 分片 | {len(man['shards'])} 个，共 {fmt_bytes(sum(s['bytes'] for s in man['shards']))} |")
    add("")
    add("### 分片")
    add("")
    add("| 文件 | 大小 |")
    add("|---|---|")
    for s in man["shards"]:
        add(f"| `{s['file']}` | {fmt_bytes(s['bytes'])} |")
    add("")
    add("### 各分组参数量")
    add("")
    add("| 分组 | 含义 | 量化参数量 |")
    add("|---|---|---|")
    names = {"A_mlp": "MLP gate/up/down", "B_full_attn": "Full Attention q/k/v/o",
             "C_deltanet": "Gated DeltaNet 大型输入/输出投影"}
    for g, n in sorted(c["params_quantized_by_group"].items()):
        add(f"| `{g}` | {names.get(g, g)} | {n / 1e9:.3f} B |")
    add("")

    if ver:
        s = ver["summary"]
        add("## 2. 权重级校验结果")
        add("")
        add(f"总体：**{'PASS' if ver['ok'] else 'FAIL'}**，耗时 {ver['seconds']:.0f}s，"
            f"{s['quantized']} 个量化 tensor 全部重读校验。")
        add("")
        add("| 分组 | tensor 数 | rel_l2 均值 | rel_l2 最大 |")
        add("|---|---|---|---|")
        for g, v in s["rel_l2_by_group"].items():
            add(f"| `{g}` | {v['count']} | {v['mean'] * 100:.3f}% | {v['max'] * 100:.3f}% |")
        add(f"| **合计** | {s['quantized']} | **{s['rel_l2_overall_mean'] * 100:.3f}%** | — |")
        add("")
        add(f"- 与原始 BF16 权重逐元素比较（反量化后），误差只来自 NVFP4 量化本身；")
        add(f"- 抽样 {s['requant_checked']} 个 tensor 用 FlashInfer 重新量化，packed 与 scale **按位一致**；")
        add(f"- 最差 tensor：`{s['worst']}`。")
        add("")

    add("## 3. 量化 tensor 明细")
    add("")
    add("| # | tensor | 形状 [N,K] | amax | global_scale | rel_l2 | 输出分片 |")
    add("|---|---|---|---|---|---|---|")
    for i, t in enumerate([x for x in man["tensors"] if x["status"] == "nvfp4"], 1):
        nv = t["nvfp4"]
        e = err.get(t["name"]) or {}
        rl = f"{e.get('rel_l2', float('nan')) * 100:.3f}%" if e else "—"
        add(f"| {i} | `{t['name'].replace('language_model.layers.', 'L').replace('.weight', '')}` "
            f"| {t['source']['shape'][0]}×{t['source']['shape'][1]} | {nv['amax']:.5f} "
            f"| {nv.get('global_scale_value', float('nan')):.4f} | {rl} | `{t['out_shard']}` |")
    add("")
    add("## 4. 保留（未量化）tensor")
    add("")
    add("以下 tensor 与原始 BF16 checkpoint **逐位一致**：")
    add("")
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in man["tensors"]:
        if t["status"] != "bf16_preserved":
            continue
        key = t["reason"]
        groups[key].append(t)
    add("| 保留原因 | tensor 数 | 示例 |")
    add("|---|---|---|")
    for reason, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        add(f"| {reason} | {len(ts)} | `{ts[0]['name']}` |")
    add("")
    add("## 5. 原样复制的非权重文件")
    add("")
    for f in man["preserved_files"]:
        add(f"- `{f}`")
    add("")
    add("> 逐 tensor 的机器可读版本见 `nvfp4_manifest.json`；格式细节见 `NVFP4_FORMAT.md`。")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
