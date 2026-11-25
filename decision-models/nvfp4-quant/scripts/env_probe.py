#!/usr/bin/env python3
"""P0 环境探针：记录量化环境的全部关键版本与能力信息。

只读取环境，不触碰 GPU 计算，不修改任何已安装包。

用法：
    python env_probe.py --out env_report.json

输出（JSON）包含：Python / PyTorch / CUDA runtime / 驱动 / GPU / Compute Capability /
FlashInfer 版本与源码路径、nvfp4_quantize 的真实签名、SfLayout 枚举、系统架构。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time


def _sha256_file(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit
        while True:
            n = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if n <= 0:
                break
            chunk = f.read(n)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception:  # pragma: no cover - 环境相关
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="env_report.json")
    ap.add_argument("--skip-gpu", action="store_true", help="只探测 Python 侧")
    args = ap.parse_args()

    probe: dict = {
        "probed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "hostname": platform.node(),
            "cpus": os.cpu_count(),
        },
        "driver": {
            "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "proc_version": None,
        },
        "container": {
            "is_container": os.path.exists("/.dockerenv"),
            "image": os.environ.get("IMAGE_NAME") or os.environ.get("NVIDIA_PYTORCH_VERSION"),
        },
        "env": {k: os.environ.get(k) for k in ("TORCH_CUDA_ARCH_LIST", "FLASHINFER_CACHE_DIR", "CUDA_HOME", "LD_LIBRARY_PATH")},
    }

    try:
        with open("/proc/driver/nvidia/version", "r", encoding="utf-8") as f:
            probe["driver"]["proc_version"] = f.read().strip().splitlines()[0]
    except Exception:
        pass

    import torch

    probe["torch"] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "debug_build": torch.version.debug,
    }
    probe["cuda_toolkit"] = {"nvcc": _run(["nvcc", "--version"])}

    if not args.skip_gpu:
        probe["gpu"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            probe["gpu"].update(
                {
                    "name": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "sm_arch_list": [f"sm_{a}{b}" for a, b in [torch.cuda.get_device_capability(0)]],
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "torch_cuda_arch_list_built": torch.cuda.get_arch_list(),
                }
            )

    try:
        import flashinfer

        probe["flashinfer"] = {
            "version": getattr(flashinfer, "__version__", "unknown"),
            "file": getattr(flashinfer, "__file__", None),
            "dir": os.path.dirname(getattr(flashinfer, "__file__", "") or ""),
        }
    except Exception as exc:  # pragma: no cover
        probe["flashinfer"] = {"error": repr(exc)}
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=2, ensure_ascii=False)
        return 1

    entry_points = {}
    try:
        from flashinfer import fp4_quantization as fq

        entry_points["module"] = fq.__file__
        entry_points["module_sha256"] = _sha256_file(fq.__file__)
        entry_points["has_nvfp4_quantize"] = hasattr(fq, "nvfp4_quantize")
        entry_points["nvfp4_quantize_signature"] = (
            str(inspect.signature(fq.nvfp4_quantize)) if hasattr(fq, "nvfp4_quantize") else None
        )
        entry_points["fp4_quantize_signature"] = (
            str(inspect.signature(fq.fp4_quantize)) if hasattr(fq, "fp4_quantize") else None
        )
        if hasattr(fq, "SfLayout"):
            entry_points["SfLayout"] = {m.name: int(m.value) for m in fq.SfLayout}
        entry_points["docstring"] = (inspect.getdoc(fq.nvfp4_quantize) or "").splitlines()[:6]
    except Exception as exc:  # pragma: no cover
        entry_points["error"] = repr(exc)
    probe["flashinfer_api"] = entry_points

    probe["pip_freeze_sha256"] = None
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    if freeze:
        probe["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
        probe["pip_freeze_core"] = [
            line for line in freeze.splitlines()
            if any(k in line.lower() for k in ("torch", "flashinfer", "triton", "nvidia-", "safetensors", "numpy"))
        ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(probe, f, indent=2, ensure_ascii=False)
    print(json.dumps(probe, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
