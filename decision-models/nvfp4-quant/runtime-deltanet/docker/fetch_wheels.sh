#!/usr/bin/env bash
# 下载构建 decision-nvfp4:0.6.18 所需的离线 wheel（放进 docker/wheels/，不入库）。
# 建议在目标机（aarch64 / Python 3.12）执行，或显式指定 --platform manylinux_2_28_aarch64。
# 可选：MIRROR=https://mirrors.aliyun.com/pypi/simple ./docker/fetch_wheels.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/wheels"
MIRROR="${MIRROR:-}"
mkdir -p "$DEST"

mirror_args=()
if [ -n "$MIRROR" ]; then
  mirror_args+=(-i "$MIRROR" --trusted-host "$(echo "$MIRROR" | awk -F/ '{print $3}')")
fi

# 1. no-deps 核心包（torch/numpy/tvm_ffi/ninja/cutlass-dsl 来自 NGC 26.07 基础镜像）
python3 -m pip download --no-deps --dest "$DEST" "${mirror_args[@]}" \
  "flashinfer-python==0.6.18" \
  "transformers==5.17.0" \
  "tokenizers==0.23.2" \
  "huggingface-hub==1.33.0" \
  "flash-linear-attention==0.6.0" \
  "fla-core==0.6.0"

# 2. FastAPI + Uvicorn 网关依赖（含纯 Python 与 aarch64 扩展）
python3 -m pip download --dest "$DEST" "${mirror_args[@]}" \
  "fastapi==0.142.2" \
  "uvicorn[standard]==0.54.0"

ls -lh "$DEST"
