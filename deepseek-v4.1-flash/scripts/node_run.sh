#!/usr/bin/env bash
# Node side, fed over ssh by serve_tp2.sh: `NAME CMD` runs CMD in a detached container NAME with GPUs,
# host network, IPC and RDMA devices, mounted kit/model/engram/cache/logs (and optional /engine, /spill),
# forwarding all TF_*, NCCL_* and TENSORFOLD_* environment variables.
#
# P0 关键修正:
#   1. 使用 `exec > >(tee /logs/$NAME.log) 2>&1; exec $CMD` 让推理进程直接作为容器 PID 1，
#      确保 `docker stop -t <T>` 发送的 SIGTERM 直达引擎以触发 Spill v2 优雅刷盘。
#   2. 支持 USE_IMAGE_ENGINE=1（使用镜像内置 /opt/tensorfold，不强制挂载宿主机 ENGINE_DIR）
#      与 USE_IMAGE_ENGINE=0（兼容旧版宿主机挂载 /engine:ro）。
#   3. 按节点挂载各自的 SPILL_DIR -> /spill 并记录 tf.recipe.spill 标签。
set -u
NAME=${1:?container name}; CMD=${2:?command}
USE_IMAGE_ENGINE=${USE_IMAGE_ENGINE:-1}
: "${IMAGE:?}" "${KIT_DIR:?}" "${MODEL_DIR:?}" "${ENGRAM_DIR:?}" "${CACHE_DIR:?}" "${LOG_DIR:?}"
mkdir -p "$CACHE_DIR" "$LOG_DIR"

# 权重目录里常用软链接指向真实权重，容器内按宿主机绝对路径解析，自动只读挂入目标目录
real=()
while IFS= read -r t; do
  [ -n "$t" ] || continue
  case "$t" in
    "$MODEL_DIR"|"$MODEL_DIR"/*|"$ENGRAM_DIR"|"$ENGRAM_DIR"/*) continue ;;
  esac
  real+=(-v "$t":"$t":ro)
done < <(find "$MODEL_DIR" "$ENGRAM_DIR" -maxdepth 2 -type l -exec readlink -f {} + 2>/dev/null | xargs -r -n1 dirname | sort -u)

engine_mount=()
if [ "$USE_IMAGE_ENGINE" != 1 ]; then
  : "${ENGINE_DIR:?}"
  engine_mount=(-v "$ENGINE_DIR:/engine:ro")
fi

spill_args=()
if [ -n "${SPILL_DIR:-}" ]; then
  mkdir -p "$SPILL_DIR"
  # 创建归属标记文件，便于优雅停服时将容器内 root 写出的 spill 文件安全交还给当前用户
  if [ "$(id -u)" != 0 ] && [ -d "$SPILL_DIR" ] && [ ! -L "$SPILL_DIR" ] && [ ! -e "$SPILL_DIR/.dsv41-tensorfold-spill" ]; then
    : > "$SPILL_DIR/.dsv41-tensorfold-spill" 2>/dev/null || true
  fi
  spill_args=(--label "tf.recipe.spill=$SPILL_DIR" -v "$SPILL_DIR:/spill")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

envs=()
while IFS= read -r v; do
  envs+=(-e "$v")
done < <(env | grep -oE '^(TF_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|TENSORFOLD_[A-Z0-9_]+)=' | tr -d =)

rdma_dev=()
[ -d /dev/infiniband ] && rdma_dev=(--device /dev/infiniband)

exec docker run -d --rm --name "$NAME" \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  "${rdma_dev[@]}" \
  "${engine_mount[@]}" \
  -v "$KIT_DIR:/kit:ro" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$ENGRAM_DIR:/engram:ro" \
  "${real[@]}" \
  "${spill_args[@]}" \
  -v "$CACHE_DIR:/cache" \
  -v "$LOG_DIR:/logs" \
  "${envs[@]}" \
  --entrypoint bash \
  "$IMAGE" -c "exec > >(tee /logs/${NAME}.log) 2>&1; exec ${CMD}"
