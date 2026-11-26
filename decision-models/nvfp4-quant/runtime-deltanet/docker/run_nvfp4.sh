#!/usr/bin/env bash
# Start the W4A4 NVFP4 decider container (one model per process).
#
#   ./docker/run_nvfp4.sh
#   PORT=8016 PPLX_MODE=isolated MODEL=pplx-decider-flash-v1.3-NVFP4 ./docker/run_nvfp4.sh
set -euo pipefail

MODELS=${MODELS:-/home/sakana/models/decision-models}
IMAGE=${IMAGE:-decision-nvfp4:0.6.18}
PORT=${PORT:-8016}
MODEL=${MODEL:-pplx-decider-flash-v1.3-NVFP4}
NAME=${NAME:-gw-flash-nvfp4}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image $IMAGE not found; build it with:"
  echo "  docker build -f docker/Dockerfile -t $IMAGE ."
  exit 1
fi

exec docker run --rm --gpus all --ipc=host --shm-size=2gb \
  --name "$NAME" \
  -p "${PORT}:8016" \
  -v "${MODELS}:/models:ro" \
  -v nvfp4-fi-cache:/opt/fi-cache \
  -e MODEL="$MODEL" \
  -e PRELOAD="${PRELOAD:-$MODEL}" \
  -e DEVICE="${DEVICE:-cuda:0}" \
  -e MAX_LOADED=1 \
  -e DECIDER_PRECISION="${DECIDER_PRECISION:-nvfp4}" \
  -e DECIDER_FP8_SCALING="${DECIDER_FP8_SCALING:-tensor}" \
  -e DECIDER_NVFP4_BACKEND="${DECIDER_NVFP4_BACKEND:-cutlass}" \
  -e DECIDER_NVFP4_PDL="${DECIDER_NVFP4_PDL:-0}" \
  -e DECIDER_PREFILL_V2="${DECIDER_PREFILL_V2:-full}" \
  -e DECIDER_PREFILL_OPT="${DECIDER_PREFILL_OPT:-0}" \
  -e DECIDER_DELTANET_OPT="${DECIDER_DELTANET_OPT:-off}" \
  -e DECIDER_NVFP4_SHARED_QUANT="${DECIDER_NVFP4_SHARED_QUANT:-0}" \
  -e PPLX_MODE="${PPLX_MODE:-isolated}" \
  -e PPLX_FAST="${PPLX_FAST:-0}" \
  "$IMAGE"
