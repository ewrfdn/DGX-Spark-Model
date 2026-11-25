#!/usr/bin/env python3
"""P0 输入检查：给原始 BF16 checkpoint 建指纹与结构清单（纯 CPU，不加载模型）。

产出 input_inventory.json：
  - 每个文件的 size / sha256（可与 release-manifest.json 对照）
  - safetensors 的 tensor 数、dtype 分布、参数量
  - 按模块前缀分组的 tensor 数 / 参数量（text backbone / vision / embed / norm ...）
  - config 关键字段（层数、layer_types、hidden/intermediate 尺寸）

用法：
  python inventory_model.py --model-dir <bf16 目录> --out input_inventory.json [--no-sha256]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import struct
import time


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def classify(key: str) -> str:
    if key.startswith("visual."):
        return "vision_encoder"
    if key.startswith("language_model.embed_tokens"):
        return "embedding"
    if key.startswith("language_model.norm"):
        return "final_norm"
    if key.startswith("lm_head"):
        return "lm_head"
    if ".mlp." in key:
        return "text_mlp"
    if ".self_attn." in key:
        return "text_full_attention"
    if ".linear_attn." in key:
        return "text_deltanet"
    if "layernorm" in key:
        return "text_norm"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/sakana/models/decision-models/pplx-decider-flash-v1.3")
    ap.add_argument("--out", default="input_inventory.json")
    ap.add_argument("--no-sha256", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    md = os.path.abspath(args.model_dir)
    inv: dict = {"model_dir": md, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    files = []
    for name in sorted(os.listdir(md)):
        p = os.path.join(md, name)
        if not os.path.isfile(p):
            continue
        rec = {"name": name, "bytes": os.path.getsize(p)}
        if not args.no_sha256:
            rec["sha256"] = sha256_file(p)
        files.append(rec)
    inv["files"] = files

    # 与官方 release-manifest.json 对照
    man_path = os.path.join(md, "release-manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
        checks = []
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "readout.safetensors", "config.json", "decision_config.json", "joint_config.json"):
            exp = (man.get("files", {}).get(name) or {}).get("sha256")
            got = next((f.get("sha256") for f in files if f["name"] == name), None)
            checks.append({"file": name, "expected": exp, "actual": got,
                           "match": (exp == got) if exp and got else None,
                           "bytes_match": (man.get("files", {}).get(name, {}).get("bytes") ==
                                           next((f["bytes"] for f in files if f["name"] == name), None))})
        inv["manifest_check"] = {"repo_id": man.get("repo_id"), "base_model": man.get("base_model"),
                                 "base_revision": man.get("base_revision"), "checks": checks}

    # safetensors 结构
    index_path = os.path.join(md, "model.safetensors.index.json")
    index = json.load(open(index_path, encoding="utf-8"))
    headers, shards = {}, sorted(set(index["weight_map"].values()))
    for s in shards + ["readout.safetensors"]:
        p = os.path.join(md, s)
        if not os.path.exists(p):
            continue
        h = {(k): v for k, v in read_safetensors_header(p).items() if k != "__metadata__"}
        headers[s] = h

    tensors, by_dtype, by_group = [], collections.Counter(), collections.defaultdict(lambda: {"count": 0, "params": 0})
    for s, h in headers.items():
        for k, v in h.items():
            n = 1
            for d in v["shape"]:
                n *= int(d)
            g = "readout" if s == "readout.safetensors" else classify(k)
            by_dtype[v["dtype"]] += 1
            by_group[g]["count"] += 1
            by_group[g]["params"] += n
            tensors.append({"name": k, "shard": s, "dtype": v["dtype"],
                            "shape": [int(x) for x in v["shape"]], "numel": n, "group": g})

    inv["config"] = json.load(open(os.path.join(md, "config.json"), encoding="utf-8"))
    tc = inv["config"].get("text_config", {})
    inv["summary"] = {
        "shards": shards,
        "tensor_count": len(tensors),
        "param_count": sum(t["numel"] for t in tensors),
        "param_count_by_group": {g: v["params"] for g, v in sorted(by_group.items())},
        "tensor_count_by_group": {g: v["count"] for g, v in sorted(by_group.items())},
        "dtype_counts": dict(by_dtype),
        "text_config": {k: tc.get(k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                               "num_attention_heads", "num_key_value_heads", "head_dim",
                                               "full_attention_interval", "linear_num_key_heads",
                                               "linear_num_value_heads", "linear_key_head_dim",
                                               "linear_value_head_dim", "tie_word_embeddings")},
        "layer_type_counts": dict(collections.Counter(tc.get("layer_types", []))),
    }
    inv["tensors"] = tensors

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)

    s = inv["summary"]
    print(f"[files] {len(files)} 个文件，权重分片 {s['shards']}")
    if inv.get("manifest_check"):
        ok = sum(1 for c in inv["manifest_check"]["checks"] if c["match"])
        print(f"[manifest] {ok}/{len(inv['manifest_check']['checks'])} 个文件 sha256 与 release-manifest 一致")
    print(f"[tensors] {s['tensor_count']} 个，总参数 {s['param_count'] / 1e9:.3f} B，dtype {s['dtype_counts']}")
    for g, n in sorted(s["param_count_by_group"].items(), key=lambda kv: -kv[1]):
        print(f"  {g:<22} {s['tensor_count_by_group'][g]:>4} tensors  {n / 1e9:.3f} B")
    print(f"[layer_types] {s['layer_type_counts']}")
    print(f"[done] {time.time() - t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
