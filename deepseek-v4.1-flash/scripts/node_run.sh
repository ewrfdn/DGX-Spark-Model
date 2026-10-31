#!/bin/bash
# Node side, fed over ssh by serve_tp2.sh (from bertholomus's TP4 recipe scripts/node_run.sh, Apache-2.0): `NAME CMD` runs CMD in a
# detached container NAME with the GPUs, host network, IPC and RDMA devices, the engine, kit, weights and cache mounted, and every TF_*,
# NCCL_* and TENSORFOLD_* variable of the caller passed in. Output goes to $LOG_DIR/NAME.log.
set -u
NAME=${1:?container name}; CMD=${2:?command}
: "${IMAGE:?}" "${ENGINE_DIR:?}" "${KIT_DIR:?}" "${MODEL_DIR:?}" "${ENGRAM_DIR:?}" "${CACHE_DIR:?}" "${LOG_DIR:?}"
mkdir -p "$CACHE_DIR" "$LOG_DIR"

# 权重目录里常用软链接指向真实权重(多套部署共用一份,省磁盘)。容器里软链接会按宿主机绝对路径去找,
# 那个路径没挂进容器就是 No such file。所以把软链接指向的目录也按原路径只读挂进去。
# 目录里没有软链接时 real=() 为空,行为和上游一致。
real=()
while IFS= read -r t; do
  [ -n "$t" ] || continue
  case "$t" in
    "$MODEL_DIR"|"$MODEL_DIR"/*|"$ENGRAM_DIR"|"$ENGRAM_DIR"/*) continue ;;  # 已经挂过了
  esac
  real+=(-v "$t":"$t":ro)
done < <(find "$MODEL_DIR" "$ENGRAM_DIR" -maxdepth 2 -type l -exec readlink -f {} + 2>/dev/null | xargs -r -n1 dirname | sort -u)

spill=()
if [ -n "${SPILL_DIR:-}" ]; then
  mkdir -p "$SPILL_DIR"
  spill=(-v "$SPILL_DIR:/spill")
fi

docker rm -f "$NAME" >/dev/null 2>&1
envs=()
while IFS= read -r v; do envs+=(-e "$v"); done < <(env | grep -oE '^(TF_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|TENSORFOLD_[A-Z0-9_]+)=' | tr -d =)
exec docker run -d --rm --name "$NAME" --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK $( [ -d /dev/infiniband ] && echo --device /dev/infiniband ) \
  -v "$ENGINE_DIR:/engine:ro" -v "$KIT_DIR:/kit:ro" -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" \
  "${real[@]}" \
  "${spill[@]}" \
  -v "$CACHE_DIR:/cache" -v "$LOG_DIR:/logs" "${envs[@]}" "$IMAGE" bash -c "( $CMD ) 2>&1 | tee /logs/$NAME.log"
