#!/usr/bin/env bash
# serve_tp2.sh start|stop|check|print [uses DSV41_CONFIG]:
# DeepSeek-V4.1-Flash 双 DGX Spark (TP2) TensorFold 启动器。
#
# P0 完整接入:
#   - 镜像内置引擎 (USE_IMAGE_ENGINE=1) 与旧版宿主机挂载 (USE_IMAGE_ENGINE=0) 双模式
#   - KV_SHARD=1 + 自动/校验 POOL (0008/0015/0018/0019/0022)
#   - Spill v2 磁盘缓存: 按节点独立 SPILL_DIR/WORKER_SPILL_DIR、大 staging 环 (256 MiB)、
#     TF_SPILL_LOAD_MS、必需指纹环境变量与可选按 rank 权重 SHA256
#   - Chunked Prefill + 增量尾巴插队 (TF_FILL_CHUNKS / TF_FILL_FIRST / TF_FILL_CUT_IN, 0041/0042/0045)
#   - TF_DS_ENGRAM_LEAD=1 (0035)、GRAPH_CAP=128 (0033)、RDMA idle nap (0010) / proxy 绑核 (0068)
#   - 优雅停服: docker stop -t <STOP_TIMEOUT> 等待 Spill 刷盘后再清理，并交还 root 写出的 spill 文件属主
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
dsv41_load_config "$here"

read -ra N <<< "$NODES"
[ "${#N[@]}" = 2 ] || { echo "NODES 必须包含两个节点 (rank 0 在前)" >&2; exit 1; }
U="http://127.0.0.1:$PORT"
PRINT=0

hca_for() { case "$1" in "${N[0]}") echo "$RDMA_HCA" ;; *) echo "${WORKER_RDMA_HCA:-$RDMA_HCA}" ;; esac; }
spill_dir_for() {
  if ! dsv41_spill_on; then echo ""; return 0; fi
  case "$1" in
    "${N[0]}") echo "$SPILL_DIR" ;;
    *) echo "${WORKER_SPILL_DIR:-$SPILL_DIR}" ;;
  esac
}
on() { ssh -o BatchMode=yes -n "$1" "$2"; }

rank() {   # rank NODE ENVS CMD
  local node=$1 extra_env=$2 cmd=$3
  local hca sdir ifname link site
  hca=$(hca_for "$node")
  sdir=$(spill_dir_for "$node")
  # 本机 nvcr 容器里 NCCL_SOCKET_IFNAME 填 CX7 接口名会触发 "no socket interface found"，默认留空让 NCCL 按路由表选择
  ifname="${NCCL_SOCKET_IF:-}"
  link="NCCL_IB_HCA=$hca"
  [ "${RDMA_RING:-1}" = 1 ] && link="$link TF_RDMA_DEVICES=$hca"
  [ -n "$ifname" ] && link="$link NCCL_SOCKET_IFNAME=$ifname"
  site="USE_IMAGE_ENGINE=$USE_IMAGE_ENGINE IMAGE=$IMAGE ENGINE_DIR=${ENGINE_DIR:-} KIT_DIR=$KIT_DIR MODEL_DIR=$MODEL_DIR ENGRAM_DIR=$ENGRAM_DIR CACHE_DIR=$CACHE_DIR LOG_DIR=$LOG_DIR SPILL_DIR=$sdir"
  if [ "$PRINT" = 1 ]; then
    echo "$node: env $extra_env $site $link node_run.sh $LANE '$cmd'"
    return 0
  fi
  ssh -o BatchMode=yes "$node" "env $extra_env $site $link bash -s -- $LANE '$cmd'" < "$here/node_run.sh" >/dev/null
}

# 停服后把容器 root 在 spill 目录写出的文件交还当前用户（仅限带 .dsv41-tensorfold-spill 标记的真实目录）
fix_spill_owner_on() {
  local node=$1 dir=$2
  [ -n "$dir" ] || return 0
  on "$node" "
    dir='$dir'; img='$IMAGE'; m='.dsv41-tensorfold-spill'
    [ -d \"\$dir\" ] && [ ! -L \"\$dir\" ] && [ -f \"\$dir/\$m\" ] && [ ! -L \"\$dir/\$m\" ] || exit 0
    me=\$(id -u); [ \"\$me\" != 0 ] || exit 0
    [ \"\$(stat -c %u \"\$dir\")\" = \"\$me\" ] && [ \"\$(stat -c %u \"\$dir/\$m\")\" = \"\$me\" ] || exit 0
    docker image inspect \"\$img\" >/dev/null 2>&1 || exit 0
    docker run --rm --network none --read-only --security-opt no-new-privileges \
      --cap-drop ALL --cap-add CHOWN --cap-add DAC_READ_SEARCH --user 0:0 \
      -v \"\$dir:/spill\" --entrypoint find \"\$img\" \
      /spill -xdev -mindepth 1 -uid 0 \\( -type f -o -type d \\) \
      -exec chown --no-dereference \"\$(id -u):\$(id -g)\" {} + >/dev/null 2>&1 || true
  " || true
}

check() {
  on "${N[0]}" "curl -sf -m 10 $U/v1/models | grep -q $SERVED_NAME && curl -sf -m 120 -H 'Content-Type: application/json' $U/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":4,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' | grep -q choices"
}

stop() {
  local timeout h
  timeout=$(dsv41_stop_timeout)
  # 先停 rank 0 (向 rank 1 发送 done 触发优雅 flush)，再停 rank 1
  for h in "${N[@]}"; do
    on "$h" "
      if docker ps -a --format '{{.Names}}' | grep -qx '$LANE'; then
        docker stop -t '$timeout' '$LANE' >/dev/null 2>&1 || true
        docker rm -f '$LANE' >/dev/null 2>&1 || true
      fi
    " || true
  done
  if dsv41_spill_on; then
    fix_spill_owner_on "${N[0]}" "$SPILL_DIR"
    fix_spill_owner_on "${N[1]}" "${WORKER_SPILL_DIR:-$SPILL_DIR}"
  fi
}

# 计算两节点各自的 TF_DS_RANK_CACHE_SHA（开启 SPILL_GIB 且 SPILL_WEIGHTS_SHA=1 时）
resolve_rank_shas() {
  RANK0_SHA=""
  RANK1_SHA=""
  if ! dsv41_spill_on || [ "${SPILL_WEIGHTS_SHA:-1}" != 1 ]; then
    return 0
  fi
  local compute=1 fn l0 l1
  [ "$PRINT" = 1 ] && compute=0
  fn=$(declare -f dsv41_rank_cache_sha_local)
  l0=$(on "${N[0]}" "$fn; dsv41_rank_cache_sha_local $compute '$CACHE_DIR/rank-cache' 0" 2>/dev/null | head -n 1 || true)
  l1=$(on "${N[1]}" "$fn; dsv41_rank_cache_sha_local $compute '$CACHE_DIR/rank-cache' 1" 2>/dev/null | head -n 1 || true)
  case "$l0" in
    sha256:*) RANK0_SHA=$l0 ;;
    PENDING)  RANK0_SHA="<computed-at-start>" ;;
    *) if [ "$PRINT" = 1 ]; then RANK0_SHA="<computed-at-start>"; else echo "FAIL: rank 0 无法计算 rank cache sha256: ${l0:-ssh 失败}" >&2; return 1; fi ;;
  esac
  case "$l1" in
    sha256:*) RANK1_SHA=$l1 ;;
    PENDING)  RANK1_SHA="<computed-at-start>" ;;
    *) if [ "$PRINT" = 1 ]; then RANK1_SHA="<computed-at-start>"; else echo "FAIL: rank 1 无法计算 rank cache sha256: ${l1:-ssh 失败}" >&2; return 1; fi ;;
  esac
}

start() {
  local eff_pool pool_set common_env spill_flags spill_env spill_ident ring fam envs r0_env r1_env
  local bin_serve bin_lanes up=0 share

  dsv41_validate_settings || return 1
  eff_pool=$(dsv41_effective_pool)
  pool_set=$(dsv41_effective_pool_set)

  # 两个 rank 共同的 P0 引擎环境变量
  common_env="$ENGINE_ENV"
  common_env="$common_env TF_RDMA_IDLE_S=$RDMA_IDLE_S TF_RDMA_NAP_US=$RDMA_NAP_US"
  [ -n "${RDMA_CPUS:-}" ] && common_env="$common_env TF_DS_RDMA_CPUS=$RDMA_CPUS"
  [ "${ENGRAM_LEAD:-1}" = 1 ] && common_env="$common_env TF_DS_ENGRAM_LEAD=1"

  # 0008: POOL / WINDOW
  if [ -n "$pool_set" ]; then
    common_env="$common_env TF_DS_POOL=$pool_set TF_DS_WINDOW=$CONTEXT"
  fi
  if [ -n "${ROUND_ROWS:-}" ] && [ "$ROUND_ROWS" != 16 ]; then
    common_env="$common_env TF_DS_ROUND_ROWS=$ROUND_ROWS"
  fi

  # 0015/0018/0019/0022: KV Sharding
  if [ "$KV_SHARD" = 1 ]; then
    common_env="$common_env TF_DS_KV_SHARD=1 TF_DS_KV_SHARD_CARRY=$KV_SHARD_CARRY TF_DS_KV_SHARD_TWINS=$KV_SHARD_TWINS TF_DS_KV_SHARD_CHECK=$KV_SHARD_CHECK"
    [ -n "${KV_SHARD_DECODE:-}" ] && common_env="$common_env TF_DS_KV_SHARD_DECODE=$KV_SHARD_DECODE"
  fi

  # 0041/0042/0045: Chunked Prefill & Cut-in
  if [ -n "${FILL_CHUNKS:-}" ] && [ "$FILL_CHUNKS" != 0 ]; then
    common_env="$common_env TF_FILL_CHUNKS=$FILL_CHUNKS"
    [ -n "${FILL_FIRST:-}" ] && [ "$FILL_FIRST" != 0 ] && common_env="$common_env TF_FILL_FIRST=$FILL_FIRST"
    [ -n "${FILL_ROUNDS:-}" ] && [ "$FILL_ROUNDS" != 0 ] && common_env="$common_env TF_FILL_ROUNDS=$FILL_ROUNDS"
    [ -n "${FILL_CUT_IN:-}" ] && [ "$FILL_CUT_IN" != 0 ] && common_env="$common_env TF_FILL_CUT_IN=$FILL_CUT_IN"
  fi

  # 0013/0014/0016/0019/0040/0059: Spill v2
  spill_flags=""
  spill_env=""
  if dsv41_spill_on; then
    share=$(awk -v q="$SPILL_QUOTA_GIB" -v g="$SPILL_GIB" 'BEGIN {printf "%.6g", q / g}')
    spill_flags="--spill-dir /spill --spill-gib $SPILL_GIB --spill-min-free-gib $SPILL_FLOOR_GIB --spill-min-tokens $SPILL_MIN_TOKENS"
    dsv41_gt0 "$SPILL_QUOTA_GIB" && spill_flags="$spill_flags --spill-quota $share"
    spill_ident=$(dsv41_spill_identity_env)
    spill_env="TF_DS_SPILL_DIR=/spill TF_DS_SPILL_GIB=$SPILL_GIB TF_DS_SPILL_MIN_FREE_GIB=$SPILL_FLOOR_GIB TF_DS_SPILL_MIN_TOKENS=$SPILL_MIN_TOKENS TF_DS_SPILL_QUOTA=$share TF_DS_SPILL_STAGING_MIB=$SPILL_STAGING_MIB TF_SPILL_TURNS=$SPILL_TURNS TF_SPILL_LOAD_MS=$SPILL_LOAD_MS TF_SPILL_LOAD_WAIT_MS=$SPILL_LOAD_WAIT_MS TF_SPILL_FLUSH_MS=$SPILL_FLUSH_MS $spill_ident"
  fi

  resolve_rank_shas || return 1

  if [ "$USE_IMAGE_ENGINE" = 1 ]; then
    bin_serve="tensorfold-native"
    bin_lanes="tf-dsv41-lanes"
  else
    bin_serve="/engine/tensorfold-native"
    bin_lanes="/engine/tf-dsv41-lanes"
  fi

  if [ "$PRINT" != 1 ]; then
    stop
    if dsv41_spill_on; then
      fix_spill_owner_on "${N[0]}" "$SPILL_DIR"
      fix_spill_owner_on "${N[1]}" "${WORKER_SPILL_DIR:-$SPILL_DIR}"
    fi
  fi

  # ---- Rank 1 (worker) ----
  ring=""
  [ "${RDMA_RING:-1}" = 1 ] && ring="--rdma $(hca_for "${N[1]}")"
  fam="--engram /engram --token-map /cache/dsv41_token_map.json --pool $eff_pool $ring --graphs $GRAPHS --side $SIDE --prefetch $PREFETCH --aio $ENGRAM_AIO --streams $STREAMS"
  [ "$GRAPHS" = 1 ] && [ -n "${GRAPH_CAP:-}" ] && fam="$fam --graph-cap $GRAPH_CAP"
  [ -n "${ROUND_ROWS:-}" ] && [ "$ROUND_ROWS" != 16 ] && fam="$fam --round-rows $ROUND_ROWS"

  r1_env="$common_env${spill_env:+ $spill_env}${RANK1_SHA:+ TF_DS_RANK_CACHE_SHA=$RANK1_SHA}"
  rank "${N[1]}" "$r1_env" "$bin_lanes /model /cache/rank-cache 1 2 $MASTER_IP $TP_PORT /kit $fam" || {
    echo "start rank 1 failed" >&2
    return 1
  }

  # ---- Rank 0 (head) ----
  envs="TF_TP_WORLD=2 TF_TP_MASTER=$MASTER_IP TF_TP_PORT=$TP_PORT TF_DS_RANK_CACHE=/cache/rank-cache TF_DS_KIT=/kit"
  envs="$envs TENSORFOLD_CUDA_KERNELS=/kit/aot TF_DS_ENGRAM=/engram TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json"
  envs="$envs TF_DS_GRAPHS=$GRAPHS TF_DS_HC_SIDE=$SIDE TF_DS_L2_PREFETCH=$PREFETCH TF_DS_ENGRAM_AIO=$ENGRAM_AIO TF_DS_STREAMS=$STREAMS"
  [ "$GRAPHS" = 1 ] && [ -n "${GRAPH_CAP:-}" ] && envs="$envs TF_DS_GRAPH_CAP=$GRAPH_CAP"
  r0_env="$envs $common_env${spill_env:+ $spill_env}${RANK0_SHA:+ TF_DS_RANK_CACHE_SHA=$RANK0_SHA}"

  local pcache_flag=""
  if [ -n "${PROMPT_CACHE_GIB:-}" ] && [[ "$SERVER_FLAGS" != *"--prompt-cache-gib"* ]]; then
    pcache_flag="--prompt-cache-gib $PROMPT_CACHE_GIB"
  fi

  rank "${N[0]}" "$r0_env" "$bin_serve serve /model --host $HOST --port $PORT --name $SERVED_NAME --context $CONTEXT --parallel $PARALLEL $pcache_flag $spill_flags $SERVER_FLAGS --backend cuda" || {
    echo "start rank 0 failed" >&2
    return 1
  }

  [ "$PRINT" = 1 ] && return 0

  for _ in $(seq 1 240); do
    sleep 5
    if on "${N[0]}" "curl -sf -m 5 $U/v1/models | grep -q $SERVED_NAME"; then up=1; break; fi
    on "${N[0]}" "docker ps --format '{{.Names}}' | grep -qx $LANE" || break
  done
  [ "$up" = 1 ] || { echo "rank 0 did not come up: see $LOG_DIR/$LANE.log on ${N[0]}" >&2; return 1; }
  check && echo "up and answering on ${N[0]}:$PORT (pool=$eff_pool kv_shard=$KV_SHARD spill=${SPILL_GIB}GiB)" || {
    echo "up, but the chat check failed" >&2
    return 1
  }
}

case "${1:?start|stop|check|print}" in
  start) start ;;
  stop) stop ;;
  check) check ;;
  print) PRINT=1; start ;;
  *) echo "start|stop|check|print" >&2; exit 2 ;;
esac
