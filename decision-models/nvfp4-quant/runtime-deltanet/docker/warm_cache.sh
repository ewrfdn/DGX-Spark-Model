#!/usr/bin/env bash
# One-time warmup of the FlashInfer/CuTe-DSL cache into the named volume.
# Run this once before the first service start (and after a driver/FlashInfer change).
set -euo pipefail
MODELS=${MODELS:-/home/sakana/models/decision-models}
IMAGE=${IMAGE:-decision-nvfp4:0.6.18}
MODEL=${MODEL:-pplx-decider-flash-v1.3-NVFP4}
TOKENS=${TOKENS:-128,560,1060,2066,4076,8092}

docker volume create nvfp4-fi-cache >/dev/null
exec docker run --rm --gpus all --ipc=host \
  -v nvfp4-fi-cache:/opt/fi-cache \
  -v "${MODELS}:/models:ro" \
  "$IMAGE" \
  python /app/scripts/warmup_cache.py --checkpoint "/models/${MODEL}" --tokens "$TOKENS"
