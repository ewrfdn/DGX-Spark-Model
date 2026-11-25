#!/usr/bin/env python3
"""pplx-decider-flash-v1.3 离线 NVFP4 权重量化（FlashInfer Max PTQ，W4A4 权重侧）。

职责边界（与验收文档一致）：
  * 只做**权重**的离线转换：BF16 → NVFP4 packed + FP8 block scale + FP32 global scale；
  * 不跑模型 forward / 不调用 Decider / 不做任何 GEMM 或 mm_fp4；
  * 不做 activation 量化、不做 QAT / 4over6 / SmoothQuant、不改模型结构；
  * 原始 BF16 checkpoint 只读，输出写到一个全新目录。

量化范围（文本 Backbone，见 config.text_config.layer_types）：
  A 组 MLP        gate_proj / up_proj / down_proj            —— 32 层
  B 组 FullAttn   q_proj / k_proj / v_proj / o_proj          —— linear_attention 为 False 的 8 层
  C 组 DeltaNet   in_proj_qkv / in_proj_z / out_proj         —— 24 层（大型输入/输出投影）
  保留            in_proj_a / in_proj_b（小型 gate/decay）、conv1d、A_log、dt_bias、
                  norm、embedding、lm/readout、vision encoder、locate-span —— 原精度

产物：
  <out>/model-0000X-of-0000Y.safetensors   量化权重 + 原精度权重
  <out>/model.safetensors.index.json       新的权重索引
  <out>/config.json                        原 config + quantization_config
  <out>/nvfp4_manifest.json                逐 tensor 量化清单（含 global scale / scale 统计）
  <out>/quantization_plan.json             允许清单与排除清单（运行前生成）
  <out>/NVFP4_FORMAT.md                    交接用格式说明
  其余非权重文件（tokenizer、readout、decision_config、locate-span 等）原样复制

用法：
  python quantize_nvfp4.py --model-dir /path/to/bf16 --out-dir /path/to/nvfp4 \
      [--shard-bytes 4.5e9] [--device cuda:0] [--plan-only] [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvfp4_ref as ref  # noqa: E402

MODEL_DIR_DEFAULT = "/home/sakana/models/decision-models/pplx-decider-flash-v1.3"
OUT_DIR_DEFAULT = "/home/sakana/models/decision-models/pplx-decider-flash-v1.3-NVFP4"

FP4 = "FLOAT4_E2M1X2"
SF_LAYOUT = "128x4_swizzled"

# ---------------------------------------------------------------------------------------
# 允许清单 / 排除清单
# ---------------------------------------------------------------------------------------
GROUP_A = re.compile(r"^language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
GROUP_B = re.compile(r"^language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
GROUP_C = re.compile(r"^language_model\.layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$")

RETAIN_RULES = [
    (re.compile(r"^language_model\.layers\.\d+\.linear_attn\.in_proj_a\.weight$"),
     "DeltaNet 小型 decay gate（32 维输出），保留 BF16"),
    (re.compile(r"^language_model\.layers\.\d+\.linear_attn\.in_proj_b\.weight$"),
     "DeltaNet 小型 beta gate（32 维输出），保留 BF16"),
    (re.compile(r"^language_model\.layers\.\d+\.linear_attn\.conv1d\.weight$"),
     "DeltaNet 短卷积核，状态相关参数，保留 BF16"),
    (re.compile(r"^language_model\.layers\.\d+\.linear_attn\.(A_log|dt_bias)$"),
     "DeltaNet decay 参数，保留 BF16"),
    (re.compile(r"^language_model\.layers\.\d+\.linear_attn\.norm\.weight$"),
     "DeltaNet gated norm，保留 BF16"),
    (re.compile(r"^language_model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight$"),
     "RMSNorm，保留 BF16"),
    (re.compile(r"^language_model\.layers\.\d+\.self_attn\.(q_norm|k_norm)\.weight$"),
     "q/k RMSNorm，保留 BF16"),
    (re.compile(r"^language_model\.embed_tokens\.weight$"), "Embedding（tie_word_embeddings），保留 BF16"),
    (re.compile(r"^language_model\.norm\.weight$"), "最终 RMSNorm，保留 BF16"),
    (re.compile(r"^visual\."), "Vision Encoder，保留 BF16"),
    (re.compile(r"^lm_head\."), "Decision / LM head，保留原精度"),
    (re.compile(r"(^|\.)readout\."), "Decision Readout，保留原精度"),
]


def explain_retained(key: str) -> str:
    for pat, reason in RETAIN_RULES:
        if pat.match(key):
            return reason
    return "不在允许清单内，保持原精度"


def build_plan(index: dict, config: dict, include_z: bool = True) -> dict:
    text_cfg = config["text_config"]
    n_layers = text_cfg["num_hidden_layers"]
    layer_types = text_cfg["layer_types"]
    assert len(layer_types) == n_layers, "layer_types 长度与 num_hidden_layers 不一致"

    weight_map = index["weight_map"]
    entries: dict[str, dict] = {}
    for key in sorted(weight_map):
        m = GROUP_A.match(key)
        if m:
            entries[key] = {"target": True, "group": "A_mlp",
                            "reason": f"MLP {m.group(2)}（第 {m.group(1)} 层）——组 A"}
            continue
        m = GROUP_B.match(key)
        if m:
            layer = int(m.group(1))
            if layer_types[layer] != "full_attention":
                raise SystemExit(f"[plan] {key} 落在非 full_attention 层（layer_types[{layer}]={layer_types[layer]}）")
            entries[key] = {"target": True, "group": "B_full_attn",
                            "reason": f"Full Attention {m.group(2)}（第 {layer} 层）——组 B"}
            continue
        m = GROUP_C.match(key)
        if m:
            layer = int(m.group(1))
            proj = m.group(2)
            if layer_types[layer] != "linear_attention":
                raise SystemExit(f"[plan] {key} 落在非 linear_attention 层（layer_types[{layer}]={layer_types[layer]}）")
            if proj == "in_proj_z" and not include_z:
                entries[key] = {"target": False, "group": "retained",
                                "reason": "--no-deltanet-z：in_proj_z 按要求保留 BF16"}
            else:
                entries[key] = {"target": True, "group": "C_deltanet",
                                "reason": f"DeltaNet {proj} 大型投影（第 {layer} 层）——组 C"}
            continue
        entries[key] = {"target": False, "group": "retained", "reason": explain_retained(key)}

    targets = [k for k, v in entries.items() if v["target"]]
    retained = [k for k, v in entries.items() if not v["target"]]
    by_group: dict[str, int] = {}
    for k in targets:
        by_group[entries[k]["group"]] = by_group.get(entries[k]["group"], 0) + 1

    return {
        "model": config.get("architectures"),
        "layer_types": layer_types,
        "counts": {"total": len(entries), "targets": len(targets), "retained": len(retained), "by_group": by_group},
        "layers": {
            "full_attention": [i for i, t in enumerate(layer_types) if t == "full_attention"],
            "linear_attention": [i for i, t in enumerate(layer_types) if t == "linear_attention"],
        },
        "entries": entries,
    }


# ---------------------------------------------------------------------------------------
# 分片写入
# ---------------------------------------------------------------------------------------
class ShardWriter:
    """把 tensor 攒到 max_bytes 再落盘；最后统一改名成 HF 风格分片名。"""

    def __init__(self, out_dir: str, max_bytes: int):
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.buf: dict[str, torch.Tensor] = {}
        self.bytes = 0
        self.tmp_paths: list[str] = []
        self.key_to_shard: dict[str, str] = {}

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.buf:
            raise KeyError(f"重复 tensor: {key}")
        self.buf[key] = tensor.contiguous()
        self.bytes += tensor.numel() * tensor.element_size()
        if self.bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        idx = len(self.tmp_paths) + 1
        tmp = os.path.join(self.out_dir, f".tmp-shard-{idx:05d}.safetensors")
        save_file(self.buf, tmp, metadata={"format": "pt", "quantized_by": "flashinfer-nvfp4-ptq"})
        for k in self.buf:
            self.key_to_shard[k] = tmp
        self.tmp_paths.append(tmp)
        print(f"  [write] {tmp}  {len(self.buf)} tensors  {self.bytes/1e9:.2f} GB", flush=True)
        self.buf = {}
        self.bytes = 0

    def finalize(self) -> tuple[list[str], dict[str, str]]:
        self.flush()
        n = len(self.tmp_paths)
        finals = []
        remap: dict[str, str] = {}
        for i, tmp in enumerate(self.tmp_paths, start=1):
            final = os.path.join(self.out_dir, f"model-{i:05d}-of-{n:05d}.safetensors")
            os.replace(tmp, final)
            finals.append(final)
            for k, v in self.key_to_shard.items():
                if v == tmp:
                    remap[k] = os.path.basename(final)
        return finals, remap


# ---------------------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------------------
@dataclass
class Stats:
    quantized: int = 0
    preserved: int = 0
    params_quantized: int = 0
    params_total: int = 0
    group_params: dict[str, int] = field(default_factory=dict)
    rel_l2: list[float] = field(default_factory=list)


def quantize_one(w: torch.Tensor, device: torch.device, nvfp4_quantize, SfLayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """把一个 [N, K] BF16 权重切成 NVFP4。返回 (packed, sf_flat, global_scale, info)。"""
    m, k = w.shape
    if k % ref.BLOCK_SIZE:
        raise ValueError(f"K={k} 不是 {ref.BLOCK_SIZE} 的倍数，无法按 block 量化")
    gpu = w.to(device=device, dtype=torch.bfloat16)
    amax = gpu.to(torch.float32).abs().amax()
    gs = ref.compute_global_scale(amax).reshape(1).to(torch.float32)
    packed, sf = nvfp4_quantize(
        gpu, gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False, sf_vec_size=ref.BLOCK_SIZE
    )
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (m, k // 2), "packed 形状/dtype 异常"
    padded_rows, padded_cols = ref.sf_padded_dims(m, k)
    assert sf.numel() == padded_rows * padded_cols, "block scale 尺寸异常"

    # scale 统计（按逻辑布局解释，用于清单；不影响写盘内容）
    sf_logical = ref.unswizzle_sf(sf, m, k).view(torch.float8_e4m3fn).to(torch.float32)
    info = {
        "amax": float(amax),
        "global_scale": float(gs.item()),
        "block_scale_min": float(sf_logical.min()),
        "block_scale_max": float(sf_logical.max()),
        "block_scale_mean": float(sf_logical.mean()),
        "block_scale_nonzero": int((sf_logical > 0).sum()),
        "sf_padded_rows": padded_rows,
        "sf_padded_cols": padded_cols,
    }
    # FlashInfer 返回的 scale 是 2D [padded_rows, padded_cols]，落盘统一压平成 1D
    return packed.cpu(), sf.reshape(-1).cpu(), gs.cpu().reshape(1), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-bytes", type=float, default=4.5e9)
    ap.add_argument("--plan-only", action="store_true", help="只生成允许清单，不量化")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 target（调试用）")
    ap.add_argument("--no-deltanet-z", action="store_true", help="in_proj_z 保留 BF16")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    model_dir = os.path.abspath(args.model_dir)
    out_dir = os.path.abspath(args.out_dir)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    config_path = os.path.join(model_dir, "config.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    plan = build_plan(index, config, include_z=not args.no_deltanet_z)
    print(f"[plan] {plan['counts']}", flush=True)

    if os.path.exists(out_dir):
        if not args.overwrite and os.listdir(out_dir):
            raise SystemExit(f"[error] 输出目录非空：{out_dir}（加 --overwrite 覆盖）")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "quantization_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    if args.plan_only:
        print(f"[plan] 已写入 {out_dir}/quantization_plan.json")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("[error] CUDA 不可用")
    from flashinfer.fp4_quantization import SfLayout, nvfp4_quantize

    device = torch.device(args.device)
    print(f"[env] gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
          f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    writer = ShardWriter(out_dir, int(args.shard_bytes))
    stats = Stats()
    manifest: list[dict] = []
    done_targets = 0

    shards = sorted(set(index["weight_map"].values()))
    for shard_name in shards:
        shard_path = os.path.join(model_dir, shard_name)
        print(f"[read] {shard_name}", flush=True)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in list(f.keys()):
                w = f.get_tensor(key)
                stats.params_total += w.numel()
                entry = plan["entries"].get(key)
                if entry is None:
                    entry = {"target": False, "group": "retained", "reason": "index 之外的新 key，保持原精度"}
                if entry["target"] and (not args.limit or done_targets < args.limit):
                    packed, sf, gs, info = quantize_one(w, device, nvfp4_quantize, SfLayout)
                    prefix = key[: -len(".weight")]
                    writer.add(f"{prefix}.weight_fp4", packed)
                    writer.add(f"{prefix}.weight_scale", sf)
                    writer.add(f"{prefix}.weight_global_scale", gs)
                    stats.quantized += 1
                    stats.params_quantized += w.numel()
                    stats.group_params[entry["group"]] = stats.group_params.get(entry["group"], 0) + w.numel()
                    manifest.append({
                        "name": key,
                        "group": entry["group"],
                        "reason": entry["reason"],
                        "status": "nvfp4",
                        "source": {"shard": shard_name, "dtype": str(w.dtype).replace("torch.", ""),
                                   "shape": list(w.shape), "numel": w.numel()},
                        "nvfp4": {
                            "weight": f"{prefix}.weight_fp4",
                            "weight_dtype": "uint8",
                            "weight_shape": list(packed.shape),
                            "packed_nibble_order": "low_nibble=even_index",
                            "scale": f"{prefix}.weight_scale",
                            "scale_dtype": "uint8(e4m3 bits)",
                            "scale_shape": [int(sf.numel())],
                            "scale_layout": SF_LAYOUT,
                            "scale_padded": [info["sf_padded_rows"], info["sf_padded_cols"]],
                            "block_size": ref.BLOCK_SIZE,
                            "global_scale": f"{prefix}.weight_global_scale",
                            "global_scale_dtype": "float32",
                            **{k: info[k] for k in ("amax", "block_scale_min", "block_scale_max",
                                                    "block_scale_mean", "block_scale_nonzero")},
                            "global_scale_value": info["global_scale"],
                        },
                    })
                    done_targets += 1
                    del packed, sf, gs
                else:
                    writer.add(key, w)
                    stats.preserved += 1
                    manifest.append({
                        "name": key,
                        "group": entry["group"],
                        "reason": entry["reason"],
                        "status": "bf16_preserved",
                        "source": {"shard": shard_name, "dtype": str(w.dtype).replace("torch.", ""),
                                   "shape": list(w.shape), "numel": w.numel()},
                    })
                del w
                done = stats.quantized + stats.preserved
                if done % 25 == 0:
                    print(f"  [progress] {done}/{len(plan['entries'])} tensors  "
                          f"quantized={stats.quantized} preserved={stats.preserved}  "
                          f"{time.time() - t_start:.0f}s", flush=True)
                if done % 100 == 0:
                    torch.cuda.empty_cache()

    finals, remap = writer.finalize()

    # 新索引：量化 tensor 落盘的是 3 个派生 key，原始 key 不再写入
    weight_map: dict[str, str] = {}
    for m in manifest:
        if m["status"] == "nvfp4":
            shard = remap[m["nvfp4"]["weight"]]
            m["out_shard"] = shard
            for sub in ("weight", "scale", "global_scale"):
                weight_map[m["nvfp4"][sub]] = shard
        else:
            m["out_shard"] = remap[m["name"]]
            weight_map[m["name"]] = m["out_shard"]
    total_bytes = sum(os.path.getsize(p) for p in finals)
    new_index = {"metadata": {"total_size": total_bytes, "format": "pt"}, "weight_map": dict(sorted(weight_map.items()))}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    # config.json：原样 + quantization_config
    qcfg = {
        "quant_method": "nvfp4",
        "format": "nvfp4_e2m1_packed_uint8 | block_scale fp8_e4m3 per-16 | global_scale fp32 per-tensor",
        "quantized_by": "flashinfer.nvfp4_quantize (Max PTQ, per-tensor global scale)",
        "weight_only": True,
        "activation_quantization": "NOT included — W4A4 activation quantization is implemented by the inference agent",
        "block_size": ref.BLOCK_SIZE,
        "global_scale_formula": "448 * 6 / amax(per-tensor)",
        "scale_layout": SF_LAYOUT,
        "scale_layout_note": "flat uint8 buffer in FlashInfer 128x4 swizzled layout (see NVFP4_FORMAT.md)",
        "nibble_order": "high nibble = even K index, low nibble = odd K index",
        "tensor_naming": "{prefix}.weight_fp4 / {prefix}.weight_scale / {prefix}.weight_global_scale",
        "manifest": "nvfp4_manifest.json",
        "groups": {"A_mlp": ["gate_proj", "up_proj", "down_proj"],
                   "B_full_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                   "C_deltanet": ["in_proj_qkv", "in_proj_z", "out_proj"]},
    }
    new_cfg = dict(config)
    new_cfg["quantization_config"] = qcfg
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)

    # 非权重文件原样复制（tokenizer / readout / decision_config / locate-span / source ...）
    copied = []
    skip = {os.path.basename(p) for p in finals} | {"model.safetensors.index.json", "config.json",
                                                    "nvfp4_manifest.json", "quantization_plan.json",
                                                    "NVFP4_FORMAT.md", "README_QUANT.md"}
    for name in sorted(os.listdir(model_dir)):
        src = os.path.join(model_dir, name)
        if name in skip or name.startswith("model-") and name.endswith(".safetensors"):
            continue
        dst = os.path.join(out_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied.append(name + "/")
        else:
            shutil.copy2(src, dst)
            copied.append(name)

    shard_info = []
    for p in finals:
        shard_info.append({"file": os.path.basename(p), "bytes": os.path.getsize(p)})

    out_manifest = {
        "schema": "pplx-decider-flash-v1.3-nvfp4-weights/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model_dir": model_dir,
        "out_dir": out_dir,
        "quantizer": {
            "library": "flashinfer",
            "api": "flashinfer.fp4_quantization.nvfp4_quantize",
            "mode": "Max PTQ (per-tensor global scale, per-16 block scale)",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "sf_layout": SF_LAYOUT,
        },
        "counts": {
            "tensors_total": len(manifest),
            "tensors_quantized": stats.quantized,
            "tensors_preserved": stats.preserved,
            "params_total": stats.params_total,
            "params_quantized": stats.params_quantized,
            "params_quantized_by_group": stats.group_params,
        },
        "shards": shard_info,
        "preserved_files": copied,
        "tensors": manifest,
    }
    with open(os.path.join(out_dir, "nvfp4_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)

    dt = time.time() - t_start
    print(f"\n[done] {dt:.1f}s  quantized={stats.quantized} preserved={stats.preserved} "
          f"params={stats.params_quantized/1e9:.3f}B  shards={len(finals)}  out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
