"""Read-only loader and validator for the offline NVFP4 checkpoint.

The checkpoint is never modified.  Validation covers the manifest, the shard
index, tensor headers (shape/dtype) and the 128x4 block-scale size; the actual
bytes are read exactly once, straight to the target device.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .linear import sf_padded_dims

SCHEMA = "pplx-decider-flash-v1.3-nvfp4-weights/v1"
#: The execution document named a separate ``quantization_config.json`` with an
#: exact field set.  The delivered artifact instead ships a descriptive
#: ``config.json:quantization_config`` plus ``nvfp4_manifest.json`` and
#: ``NVFP4_FORMAT.md``.  We validate the *semantics* of the delivered artifact
#: and record the contract delta, instead of pretending the file exists.
DOCUMENTED_QUANT_CONFIG = {
    "format": "decider_flashinfer_nvfp4_weights_v1",
    "format_version": 1,
    "artifact_type": "quantized_weights_only",
    "quant_method": "nvfp4_max_ptq",
    "weight_dtype": "fp4_e2m1",
    "packed_storage_dtype": "uint8",
    "block_scale_dtype": "float8_e4m3fn",
    "global_scale_dtype": "float32",
    "global_scale_convention": "quant_multiplier",
    "group_size": 16,
    "block_axis": "K",
    "scale_layout": "128x4",
    "do_shuffle": False,
    "weight_orientation": "N,K",
    "target_arch": "sm_121",
    "activation_calibration": None,
}
SUFFIX_FP4 = "weight_fp4"
SUFFIX_SCALE = "weight_scale"
SUFFIX_GS = "weight_global_scale"
MAP_SUFFIX = {
    SUFFIX_FP4: "weight_packed",
    SUFFIX_SCALE: "weight_block_scale",
}


@dataclass(frozen=True)
class QuantizedModule:
    prefix: str
    group: str
    n: int
    k: int
    packed_key: str
    scale_key: str
    global_scale_key: str
    scale_padded: tuple[int, int]

    def model_keys(self) -> dict[str, str]:
        """Checkpoint key -> ``NVFP4Linear`` buffer name."""
        return {
            self.packed_key: f"{self.prefix}.weight_packed",
            self.scale_key: f"{self.prefix}.weight_block_scale",
            self.global_scale_key: f"{self.prefix}.weight_global_scale",
        }


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class Checkpoint:
    """The NVFP4 checkpoint directory (weights only; the Decider code lives elsewhere)."""

    def __init__(self, root: str | Path, *, verify_sha256: bool = False) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"NVFP4 checkpoint not found: {self.root}")
        self.manifest = json.loads((self.root / "nvfp4_manifest.json").read_text())
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.config = json.loads((self.root / "config.json").read_text())
        self.quant_config: dict[str, Any] = {}
        qpath = self.root / "quantization_config.json"
        if qpath.exists():
            self.quant_config = json.loads(qpath.read_text())
        elif isinstance(self.config.get("quantization_config"), dict):
            self.quant_config = self.config["quantization_config"]
        self.verify_sha256_enabled = verify_sha256
        self.sha256: dict[str, str] = {}
        self.problems: list[str] = []

        self.weight_map: dict[str, str] = self.index["weight_map"]
        self.shards = sorted(set(self.weight_map.values()))
        self.quantized: dict[str, QuantizedModule] = {}
        self.preserved: list[str] = []
        self._parse_manifest()

    # ------------------------------------------------------------------ parse

    def _parse_manifest(self) -> None:
        entries = self.manifest.get("tensors")
        if not isinstance(entries, list):
            raise ValueError("nvfp4_manifest.json has no 'tensors' list")
        for entry in entries:
            name = entry["name"]
            status = entry.get("status")
            if status == "bf16_preserved":
                self.preserved.append(name)
                continue
            if status != "nvfp4":
                raise ValueError(f"manifest entry {name!r} has unknown status {status!r}")
            nv = entry.get("nvfp4") or {}
            shape = entry["source"]["shape"]
            if len(shape) != 2:
                raise ValueError(f"{name}: source shape {shape} is not 2-D")
            n, k = int(shape[0]), int(shape[1])
            self.quantized[name.removesuffix(".weight")] = QuantizedModule(
                prefix=name.removesuffix(".weight"),
                group=entry.get("group", ""),
                n=n,
                k=k,
                packed_key=nv["weight"],
                scale_key=nv["scale"],
                global_scale_key=nv["global_scale"],
                scale_padded=tuple(int(x) for x in nv["scale_padded"]),
            )

    # --------------------------------------------------------------- validate

    def validate(self, *, check_headers: bool = True) -> dict[str, Any]:
        """Structural validation; raises ``ValueError`` with every problem found."""
        problems: list[str] = []
        report: dict[str, Any] = {
            "schema": self.manifest.get("schema"),
            "tensors_manifest": len(self.manifest.get("tensors", [])),
            "quantized": len(self.quantized),
            "preserved": len(self.preserved),
            "shards": [],
            "problems": problems,
        }
        if self.manifest.get("schema") != SCHEMA:
            problems.append(f"manifest schema {self.manifest.get('schema')!r} != {SCHEMA!r}")

        counts = self.manifest.get("counts", {})
        if counts.get("tensors_total") != len(self.manifest.get("tensors", [])):
            problems.append("counts.tensors_total does not match len(tensors)")
        if counts.get("tensors_quantized") != len(self.quantized):
            problems.append("counts.tensors_quantized does not match manifest")
        if counts.get("tensors_preserved") != len(self.preserved):
            problems.append("counts.tensors_preserved does not match manifest")

        contract = self.contract_report()
        report["contract"] = contract
        if not contract["ok"]:
            problems.extend(contract["problems"])

        # Every manifest name must be represented in the index: preserved as-is,
        # quantized by exactly its three packed keys.
        expected_keys: set[str] = set(self.preserved)
        for module in self.quantized.values():
            expected_keys.update(module.model_keys().keys())
        index_keys = set(self.weight_map)
        missing = sorted(expected_keys - index_keys)
        unexpected = sorted(index_keys - expected_keys)
        if missing:
            problems.append(f"index is missing {len(missing)} keys, e.g. {missing[:3]}")
        if unexpected:
            problems.append(f"index has {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

        metadata = self.index.get("metadata", {})
        total_size = metadata.get("total_size")
        file_bytes = sum((self.root / shard).stat().st_size for shard in self.shards if (self.root / shard).exists())
        if total_size is not None and total_size != file_bytes:
            problems.append(f"index total_size {total_size} != shard bytes {file_bytes}")
        for shard in self.shards:
            if not (self.root / shard).exists():
                problems.append(f"missing shard {shard}")
        report["shards"] = [
            {"file": s, "bytes": (self.root / s).stat().st_size, "sha256": None} for s in self.shards
        ]
        report["total_bytes"] = file_bytes

        # Logical layout contract.
        for module in self.quantized.values():
            if module.n % 64 or module.k % 64:
                problems.append(f"{module.prefix}: N={module.n}, K={module.k} not multiples of 64")
            pr, pc = sf_padded_dims(module.n, module.k)
            if module.scale_padded != (pr, pc):
                problems.append(
                    f"{module.prefix}: manifest scale_padded {module.scale_padded} != expected {(pr, pc)}"
                )

        if check_headers:
            report["header_problems"] = self._check_headers()
            problems.extend(report["header_problems"])

        if self.verify_sha256_enabled:
            for shard in self.shards:
                path = self.root / shard
                digest = sha256_file(path)
                self.sha256[shard] = digest
                report["shards"][self.shards.index(shard)]["sha256"] = digest

        report["ok"] = not problems
        if problems:
            raise ValueError("NVFP4 checkpoint validation failed:\n  - " + "\n  - ".join(problems))
        return report

    def contract_report(self) -> dict[str, Any]:
        """Check the delivered quant-config semantics against the V1 contract.

        The V1 document specified ``quantization_config.json``; this artifact
        instead documents the same facts in ``config.json:quantization_config``
        (descriptive strings), ``nvfp4_manifest.json`` and ``NVFP4_FORMAT.md``.
        """
        q = self.quant_config
        problems: list[str] = []
        notes: list[str] = []
        documented_file = (self.root / "quantization_config.json").exists()
        if not documented_file:
            notes.append(
                "quantization_config.json is absent; validation uses config.json:quantization_config, "
                "nvfp4_manifest.json and NVFP4_FORMAT.md (recorded deviation from the V1 file contract)"
            )

        fmt = str(q.get("format", ""))
        if "nvfp4" not in fmt.lower():
            problems.append(f"quantization format {fmt!r} does not declare NVFP4")
        method = str(q.get("quant_method", ""))
        if method not in {"nvfp4", "nvfp4_max_ptq", "modelopt"}:
            problems.append(f"quant_method {method!r} is not a recognised NVFP4 Max-PTQ method")
        block = q.get("block_size", q.get("group_size"))
        if block != 16:
            problems.append(f"block_size/group_size {block!r} != 16")
        layout = str(q.get("scale_layout", ""))
        if not layout.lower().startswith("128x4"):
            problems.append(f"scale_layout {layout!r} is not 128x4")
        if q.get("do_shuffle") not in (None, False):
            problems.append(f"do_shuffle={q.get('do_shuffle')!r} != false")
        if q.get("activation_calibration") not in (None,):
            problems.append("activation_calibration must be null for W4A4 weight-only artifacts")
        if q.get("weight_only") is False:
            problems.append("artifact reports weight_only=false, incompatible with a weights-only contract")
        gs_formula = str(q.get("global_scale_formula", ""))
        if gs_formula and "448" not in gs_formula and "2688" not in gs_formula:
            problems.append(f"unexpected global_scale_formula {gs_formula!r}")

        # The descriptive config's nibble_order string contradicts the manifest
        # and NVFP4_FORMAT.md.  The manifest + measured numerics are authoritative.
        nibble = str(q.get("nibble_order", ""))
        if nibble and "high nibble = even" in nibble.lower():
            notes.append(
                "config.json nibble_order says 'high nibble = even K index', which contradicts "
                "nvfp4_manifest.json (packed_nibble_order=low_nibble=even_index) and NVFP4_FORMAT.md; "
                "the manifest/format description and a bit-level GEMM check are authoritative"
            )

        quantizer = self.manifest.get("quantizer", {})
        cap = quantizer.get("compute_capability")
        if cap is not None and list(cap) != [12, 1]:
            problems.append(f"manifest quantizer compute_capability {cap} != [12, 1]")
        if quantizer.get("sf_layout") not in (None, "128x4_swizzled"):
            problems.append(f"manifest sf_layout {quantizer.get('sf_layout')!r} != '128x4_swizzled'")

        return {
            "ok": not problems,
            "documented_file": documented_file,
            "fields": {
                "format": fmt,
                "quant_method": method,
                "block_size": block,
                "scale_layout": layout,
                "weight_only": q.get("weight_only"),
                "target_arch": q.get("target_arch") or self.manifest.get("quantizer", {}).get("compute_capability"),
            },
            "notes": notes,
            "problems": problems,
        }

    def _check_headers(self) -> list[str]:
        problems: list[str] = []
        by_shard: dict[str, list[str]] = {}
        for key, shard in self.weight_map.items():
            by_shard.setdefault(shard, []).append(key)

        expected: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for key in self.preserved:
            expected[key] = (None, None)  # filled from manifest source shape/dtype
        for module in self.quantized.values():
            expected[module.packed_key] = ((module.n, module.k // 2), torch.uint8)
            expected[module.scale_key] = ((module.scale_padded[0] * module.scale_padded[1],), torch.uint8)
            expected[module.global_scale_key] = ((1,), torch.float32)

        # Source shapes/dtypes for preserved tensors come from the manifest.
        manifest_dtype = {
            entry["name"]: entry["source"]["dtype"] for entry in self.manifest["tensors"]
        }
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
            "uint8": torch.uint8,
        }
        for key in self.preserved:
            expected[key] = (None, dtype_map.get(manifest_dtype[key], torch.bfloat16))

        for shard, keys in by_shard.items():
            path = self.root / shard
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                want = set(keys)
                if actual_keys != want:
                    problems.append(
                        f"{shard}: header keys {len(actual_keys)} != index {len(want)} "
                        f"(missing {len(want - actual_keys)}, extra {len(actual_keys - want)})"
                    )
                for key in keys:
                    if key not in actual_keys:
                        continue
                    shape, dtype = expected.get(key, (None, None))
                    sl = handle.get_slice(key)
                    got_shape = tuple(sl.get_shape())
                    got_dtype = sl.get_dtype()
                    if shape is not None and got_shape != shape:
                        problems.append(f"{key}: shape {got_shape} != expected {shape}")
                    if dtype is not None and got_dtype != str(dtype).split(".")[-1]:
                        # safetensors reports e.g. 'BF16'; normalise
                        if not _dtype_matches(got_dtype, dtype):
                            problems.append(f"{key}: dtype {got_dtype} != expected {dtype}")
        return problems

    # -------------------------------------------------------------------- load

    def load_state_dict(self, device: str | torch.device = "cuda", *, dtype: torch.dtype | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Load all tensors straight to ``device`` and rename packed weights.

        Returns ``(state_dict, stats)``.  Quantised weights are never expanded.
        """
        started = time.perf_counter()
        state: dict[str, torch.Tensor] = {}
        seen: set[str] = set()
        for shard in self.shards:
            path = self.root / shard
            with safe_open(str(path), framework="pt", device=str(device)) as handle:
                for key in handle.keys():
                    if key in seen:
                        raise ValueError(f"duplicate tensor {key!r} across shards")
                    seen.add(key)
                    tensor = handle.get_tensor(key)
                    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    state[self._model_key(key)] = tensor
        missing = set(self.weight_map) - seen
        if missing:
            raise ValueError(f"shards are missing {len(missing)} indexed keys, e.g. {sorted(missing)[:3]}")
        stats = {
            "seconds": round(time.perf_counter() - started, 3),
            "tensors": len(state),
            "bytes": sum(t.numel() * t.element_size() for t in state.values()),
            "device": str(device),
        }
        return state, stats

    def _model_key(self, key: str) -> str:
        if key.endswith("." + SUFFIX_FP4):
            return key.removesuffix("." + SUFFIX_FP4) + ".weight_packed"
        if key.endswith("." + SUFFIX_SCALE):
            return key.removesuffix("." + SUFFIX_SCALE) + ".weight_block_scale"
        return key

    # -------------------------------------------------------------------- misc

    def summary(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for module in self.quantized.values():
            groups[module.group] = groups.get(module.group, 0) + 1
        return {
            "root": str(self.root),
            "schema": self.manifest.get("schema"),
            "quantized": len(self.quantized),
            "quantized_by_group": groups,
            "preserved": len(self.preserved),
            "shards": self.shards,
            "total_bytes": sum((self.root / s).stat().st_size for s in self.shards),
            "sha256": self.sha256,
        }


def _dtype_matches(reported: str, expected: torch.dtype) -> bool:
    name = reported.replace("torch.", "").upper()
    table = {
        torch.bfloat16: {"BF16", "BFLOAT16"},
        torch.float32: {"F32", "FLOAT32"},
        torch.float16: {"F16", "FLOAT16"},
        torch.uint8: {"U8", "UINT8"},
    }
    return name in table.get(expected, set())
