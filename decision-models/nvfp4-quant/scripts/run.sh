#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# NVFP4 权重量化流水线（容器内执行，与生产推理环境隔离）
#
#   ./run.sh probe          # P0：环境探针 -> out/env_report.json
#   ./run.sh smoke          # P0：nvfp4_quantize smoke test -> out/smoke_report.json
#   ./run.sh inventory      # P0：原始 checkpoint 指纹/结构清单 -> out/input_inventory.json
#   ./run.sh plan           # 只生成允许/排除清单 -> out/quantization_plan.json
#   ./run.sh quantize       # 正式量化 -> out/ 下的 NVFP4 checkpoint
#   ./run.sh verify         # 权重级校验 -> out/verify_report.json
#   ./run.sh all            # probe + smoke + plan + quantize + verify
#
# 环境变量（都有默认值）：
#   MODEL_DIR  原始 BF16 checkpoint（只读挂载）
#   OUT_DIR    量化产物目录
#   IMAGE      量化镜像（默认 nvfp4-quant:0.6.3）
#   CACHE      FlashInfer JIT 编译缓存（跨次复用，默认 ./.fi-cache）
# ---------------------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/sakana/models/decision-models/pplx-decider-flash-v1.3}"
OUT_DIR="${OUT_DIR:-/home/sakana/models/decision-models/pplx-decider-flash-v1.3-NVFP4}"
IMAGE="${IMAGE:-nvfp4-quant:0.6.3}"
CACHE="${CACHE:-${HERE}/../.fi-cache}"

mkdir -p "$OUT_DIR" "$CACHE"

docker_run() {
  docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${OUT_DIR}:/out" \
    -v "${CACHE}:/opt/fi-cache" \
    -v "${HERE}/scripts:/scripts:ro" \
    -w /scripts \
    "$IMAGE" "$@"
}

cmd="${1:-help}"
case "$cmd" in
  probe)
    docker_run python env_probe.py --out /out/env_report.json
    ;;
  smoke)
    docker_run python smoke_test_nvfp4.py --out /out/smoke_report.json
    ;;
  inventory)
    # 纯 CPU：只读 safetensors header + 文件 sha256，不加载权重
    docker run --rm -v "${MODEL_DIR}:/model:ro" -v "${OUT_DIR}:/out" \
      -v "${HERE}/scripts:/scripts:ro" -w /scripts \
      python:3.12-slim python inventory_model.py --model-dir /model --out /out/input_inventory.json
    ;;
  plan)
    docker_run python quantize_nvfp4.py --model-dir /model --out-dir /out --plan-only
    ;;
  quantize)
    shift || true
    docker_run python quantize_nvfp4.py --model-dir /model --out-dir /out "$@"
    ;;
  verify)
    docker_run python verify_nvfp4_ckpt.py --model-dir /model --ckpt-dir /out --out /out/verify_report.json
    ;;
  docs)
    docker_run python make_manifest_md.py --manifest /out/nvfp4_manifest.json \
      --verify /out/verify_report.json --out /out/量化清单.md
    ;;
  all)
    "$0" probe
    "$0" smoke
    "$0" plan
    "$0" quantize --overwrite
    "$0" verify
    "$0" docs
    ;;
  shell)
    docker run --rm -it --gpus all --ipc=host \
      -v "${MODEL_DIR}:/model:ro" -v "${OUT_DIR}:/out" \
      -v "${CACHE}:/opt/fi-cache" -v "${HERE}/scripts:/scripts:ro" \
      -w /scripts "$IMAGE" bash
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
