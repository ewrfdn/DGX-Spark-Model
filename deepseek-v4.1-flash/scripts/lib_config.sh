#!/usr/bin/env bash
# lib_config.sh: 统一配置加载与 P0 KV-Sharding / Pool / Spill v2 参数解析与校验。
# 配置解析优先级: $DSV41_CONFIG -> <dir>/config.tp2.env -> <dir>/local.env

dsv41_resolve_config() {
  local dir=${1:?script dir}
  if [ -n "${DSV41_CONFIG:-}" ]; then
    echo "$DSV41_CONFIG"
  elif [ -r "$dir/config.tp2.env" ]; then
    echo "$dir/config.tp2.env"
  elif [ -r "$dir/local.env" ]; then
    echo "$dir/local.env"
  else
    echo "$dir/local.env"
  fi
}

dsv41_load_config() {
  local dir=${1:?script dir}
  local cfg
  cfg=$(dsv41_resolve_config "$dir")
  if [ ! -r "$cfg" ]; then
    echo "没有配置文件 $cfg (先 cp config/tp2.env.example scripts/local.env 或设置 DSV41_CONFIG)" >&2
    exit 1
  fi
  CFG="$cfg"
  # shellcheck disable=SC1090
  source "$CFG"

  # ---- 运行模式与上游版本 pin 默认值 ----
  USE_IMAGE_ENGINE="${USE_IMAGE_ENGINE:-1}"
  BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
  TF_REPO="${TF_REPO:-https://github.com/bertholomus/TensorFold.git}"
  TF_REF="${TF_REF:-fbbc97132cd004237d9887b8d3185896efaedc5c}"
  INPUTS_REF="${INPUTS_REF:-59402086381a42c74d53053443224eec2790b4c5}"
  MIAAI_REPO="${MIAAI_REPO:-https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git}"
  MIAAI_REF="${MIAAI_REF:-c7f2b85100f0a7b099c92a4e21825e30d93c9c51}"
  MODEL_ID="${MODEL_ID:-Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw}"
  MODEL_REVISION="${MODEL_REVISION:-45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b}"
  MODEL_QUANT="${MODEL_QUANT:-exl3-2.9bpw}"
  ENGRAM_ID="${ENGRAM_ID:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
  ENGRAM_REVISION="${ENGRAM_REVISION:-1bae0e1088993b9b82de08f59de5be5a34e42170}"

  # ---- 引擎运行与图/通信参数 ----
  CONTEXT="${CONTEXT:-1048576}"
  POOL="${POOL:-}"
  PARALLEL="${PARALLEL:-4}"
  STREAMS="${STREAMS:-$PARALLEL}"
  ROUND_ROWS="${ROUND_ROWS:-16}"
  GRAPHS="${GRAPHS:-1}"
  GRAPH_CAP="${GRAPH_CAP:-128}"
  SIDE="${SIDE:-1}"
  PREFETCH="${PREFETCH:-1}"
  ENGRAM_AIO="${ENGRAM_AIO:-1}"
  ENGRAM_LEAD="${ENGRAM_LEAD:-1}"
  RDMA_RING="${RDMA_RING:-1}"
  RDMA_IDLE_S="${RDMA_IDLE_S:-2}"
  RDMA_NAP_US="${RDMA_NAP_US:-500}"
  RDMA_CPUS="${RDMA_CPUS:-}"
  PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"

  # ---- Chunked Prefill + 增量尾巴插队 (0041/0042/0045) ----
  FILL_CHUNKS="${FILL_CHUNKS:-4}"
  FILL_FIRST="${FILL_FIRST:-4}"
  FILL_ROUNDS="${FILL_ROUNDS:-}"
  FILL_CUT_IN="${FILL_CUT_IN:-8192}"

  # ---- KV Sharding (0008/0015/0018/0019/0022) ----
  KV_SHARD="${KV_SHARD:-0}"
  KV_SHARD_DECODE="${KV_SHARD_DECODE:-}"
  KV_SHARD_CARRY="${KV_SHARD_CARRY:-1}"
  KV_SHARD_TWINS="${KV_SHARD_TWINS:-1}"
  KV_SHARD_CHECK="${KV_SHARD_CHECK:-1024}"

  # ---- Spill v2 磁盘 KV 缓存 (0013/0014/0016/0019/0040/0059) ----
  SPILL_GIB="${SPILL_GIB:-0}"
  SPILL_DIR="${SPILL_DIR:-}"
  WORKER_SPILL_DIR="${WORKER_SPILL_DIR:-$SPILL_DIR}"
  SPILL_FLOOR_GIB="${SPILL_FLOOR_GIB:-50}"
  SPILL_QUOTA_GIB="${SPILL_QUOTA_GIB:-0}"
  SPILL_MIN_TOKENS="${SPILL_MIN_TOKENS:-8192}"
  SPILL_STAGING_MIB="${SPILL_STAGING_MIB:-256}"
  SPILL_TURNS="${SPILL_TURNS:-4}"
  SPILL_LOAD_MS="${SPILL_LOAD_MS:-120}"
  SPILL_LOAD_WAIT_MS="${SPILL_LOAD_WAIT_MS:-30000}"
  SPILL_FLUSH_MS="${SPILL_FLUSH_MS:-60000}"
  SPILL_WEIGHTS_SHA="${SPILL_WEIGHTS_SHA:-1}"
}

dsv41_is_num() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
dsv41_gt0()    { awk -v x="$1" 'BEGIN {exit !(x + 0 > 0)}'; }
dsv41_spill_on() { dsv41_is_num "${SPILL_GIB:-0}" && dsv41_gt0 "${SPILL_GIB:-0}"; }

# KV_SHARD=1 时自动推导满足两份完整 CONTEXT 窗口 + (ROUND_ROWS + 2) 且 2048 对齐的最小 POOL:
#   CONTEXT=1048576, ROUND_ROWS=16 -> 2101248
#   CONTEXT=524288,  ROUND_ROWS=16 -> 1052672
dsv41_shard_min_pool() {
  local rows="${ROUND_ROWS:-16}"
  local n=$(( CONTEXT + rows + 2 ))
  echo $(( (n + 2047) / 2048 * 2048 * 2 ))
}

# 返回需显式下发给 TF_DS_POOL 的值（为空表示沿用 CONTEXT）
dsv41_effective_pool_set() {
  if [ -n "${POOL:-}" ]; then
    echo "$POOL"
  elif [ "${KV_SHARD:-0}" = 1 ]; then
    dsv41_shard_min_pool
  else
    echo ""
  fi
}

dsv41_effective_pool() {
  local p
  p=$(dsv41_effective_pool_set)
  echo "${p:-$CONTEXT}"
}

dsv41_stop_timeout() {
  if [ -n "${STOP_TIMEOUT:-}" ]; then
    echo "$STOP_TIMEOUT"
  elif dsv41_spill_on; then
    local ms="${SPILL_FLUSH_MS:-60000}"
    [[ "$ms" =~ ^[0-9]+$ ]] || ms=60000
    echo $(( (ms + 999) / 1000 + 30 ))
  else
    echo 30
  fi
}

# 校验 POOL / KV_SHARD / SPILL 数值合法性（不访问网络）
dsv41_validate_settings() {
  local pool need part usable rows
  rows="${ROUND_ROWS:-16}"
  [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: CONTEXT 非法: $CONTEXT" >&2; return 1; }
  (( CONTEXT <= 1048576 )) || { echo "FAIL: CONTEXT ($CONTEXT) 超过模型最大窗口 1048576" >&2; return 1; }
  [[ "$KV_SHARD" =~ ^[01]$ ]] || { echo "FAIL: KV_SHARD 必须为 0 或 1，当前为 $KV_SHARD" >&2; return 1; }
  [[ -z "$KV_SHARD_DECODE" || "$KV_SHARD_DECODE" =~ ^(fetch|owner|auto)$ ]] || {
    echo "FAIL: KV_SHARD_DECODE 必须为空 / fetch / owner / auto，当前为 $KV_SHARD_DECODE" >&2; return 1;
  }
  [[ -z "$POOL" || "$POOL" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: POOL 必须为空或正整数，当前为 $POOL" >&2; return 1; }

  pool=$(dsv41_effective_pool)
  (( pool >= CONTEXT )) || { echo "FAIL: POOL ($pool) 小于 CONTEXT ($CONTEXT)" >&2; return 1; }
  (( pool <= 16 * CONTEXT )) || { echo "FAIL: POOL ($pool) 超过 16x CONTEXT ($CONTEXT)，内存必然不足" >&2; return 1; }

  if [ "$KV_SHARD" = 1 ]; then
    need=$(dsv41_shard_min_pool)
    if (( pool % 4096 != 0 )); then
      echo "FAIL: KV_SHARD=1 要求 POOL ($pool) 为 4096 的整数倍（建议 POOL=$need 或留空自动推导）" >&2
      return 1
    fi
    part=$(( pool / 2 ))
    usable=$(( part - rows - 2 ))
    if (( usable < CONTEXT )); then
      echo "FAIL: KV_SHARD=1 下 POOL=$pool 切分后单份仅可容纳 $usable tokens < CONTEXT ($CONTEXT)；请设 POOL=$need 或留空" >&2
      return 1
    fi
  fi

  if [ "${GRAPHS:-1}" = 1 ] && (( pool % 4096 != 0 )); then
    echo "WARN: GRAPHS=1 且 POOL=$pool 不是 4096 的整数倍，深上下文可能触发 MissingTritonVariant" >&2
  fi

  if dsv41_spill_on; then
    [ -n "${SPILL_DIR:-}" ] || { echo "FAIL: 开启 SPILL_GIB=$SPILL_GIB 时必须设置 SPILL_DIR" >&2; return 1; }
    [ -n "${WORKER_SPILL_DIR:-}" ] || { echo "FAIL: 开启 SPILL_GIB=$SPILL_GIB 时必须设置 WORKER_SPILL_DIR" >&2; return 1; }
    [[ "$SPILL_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "FAIL: SPILL_DIR 必须为绝对路径且不含特殊字符: $SPILL_DIR" >&2; return 1; }
    [[ "$WORKER_SPILL_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "FAIL: WORKER_SPILL_DIR 必须为绝对路径且不含特殊字符: $WORKER_SPILL_DIR" >&2; return 1; }
    dsv41_is_num "$SPILL_FLOOR_GIB" || { echo "FAIL: SPILL_FLOOR_GIB 非法: $SPILL_FLOOR_GIB" >&2; return 1; }
    dsv41_is_num "$SPILL_QUOTA_GIB" || { echo "FAIL: SPILL_QUOTA_GIB 非法: $SPILL_QUOTA_GIB" >&2; return 1; }
    awk -v q="$SPILL_QUOTA_GIB" -v g="$SPILL_GIB" 'BEGIN {exit !(q + 0 <= g + 0)}' || {
      echo "FAIL: SPILL_QUOTA_GIB ($SPILL_QUOTA_GIB) 不能大于 SPILL_GIB ($SPILL_GIB)" >&2; return 1;
    }
    [[ "$SPILL_STAGING_MIB" =~ ^[0-9]+$ ]] && (( SPILL_STAGING_MIB >= 16 )) || {
      echo "FAIL: SPILL_STAGING_MIB 必须为 >= 16 的整数，当前为 $SPILL_STAGING_MIB" >&2; return 1;
    }
    [[ "$SPILL_MIN_TOKENS" =~ ^[0-9]+$ ]] || { echo "FAIL: SPILL_MIN_TOKENS 非法: $SPILL_MIN_TOKENS" >&2; return 1; }
    [[ "$SPILL_LOAD_MS" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: SPILL_LOAD_MS 必须为正整数，当前为 $SPILL_LOAD_MS" >&2; return 1; }
    [[ "$SPILL_LOAD_WAIT_MS" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: SPILL_LOAD_WAIT_MS 必须为正整数，当前为 $SPILL_LOAD_WAIT_MS" >&2; return 1; }
    [[ "$SPILL_FLUSH_MS" =~ ^[0-9]+$ ]] || { echo "FAIL: SPILL_FLUSH_MS 必须为非负整数，当前为 $SPILL_FLUSH_MS" >&2; return 1; }
  fi
  return 0
}

# 读取本地镜像 label（不存在时返回空）
dsv41_image_label() {
  local img=$1 key=$2 v=""
  v=$(docker image inspect -f "{{index .Config.Labels \"$key\"}}" "$img" 2>/dev/null || true)
  [ "$v" = "<no value>" ] && v=""
  echo "$v"
}

# 生成 Spill v2 必需的两机一致身份变量 (rank_identity.Extra.env)
dsv41_spill_identity_env() {
  dsv41_spill_on || return 0
  local patches ref eid mrev erev
  patches=$(dsv41_image_label "$IMAGE" tf.patches)
  [ -n "$patches" ] || patches="${PATCHES_HASH:-miaai-${MIAAI_REF:0:12}}"
  ref=$(dsv41_image_label "$IMAGE" tf.ref)
  [ -n "$ref" ] || ref="$TF_REF"
  eid="${TF_DS_ENGINE_ID:-tensorfold@${ref:0:12}+${patches}}"
  mrev="${TF_DS_MODEL_REVISION:-${MODEL_ID}@${MODEL_REVISION},quant=${MODEL_QUANT},ablit=0}"
  erev="${TF_DS_ENGRAM_REVISION:-${ENGRAM_ID}@${ENGRAM_REVISION}}"
  printf 'TF_DS_ENGINE_ID=%s TF_DS_RECIPE_PATCHES=%s TF_DS_MODEL_REVISION=%s TF_DS_ENGRAM_REVISION=%s' \
    "$eid" "${TF_DS_RECIPE_PATCHES:-$patches}" "$mrev" "$erev"
}

# 计算并缓存单节点 rank cache 文件的 SHA-256 (O_DIRECT 避免污染 page cache)
# 输出: sha256:<64hex> 或 ERR <原因>
dsv41_rank_cache_sha_local() {
  local compute=$1 dir=$2 rank=$3 f="" n=0 x st st2 a b c sum cache
  for x in "$dir"/rank"$rank"of2-*.bin; do
    if [ -f "$x" ] && [ ! -L "$x" ]; then f=$x; n=$((n + 1)); fi
  done
  [ "$n" = 1 ] || { echo "ERR found $n files matching rank${rank}of2-*.bin in $dir"; return 0; }
  st=$(stat -c '%s %Y %i' -- "$f" 2>/dev/null) || { echo "ERR cannot stat $f"; return 0; }
  cache="$dir/.$(basename -- "$f").sha256"
  if [ -r "$cache" ] && read -r a b c sum < "$cache" && [ "$a $b $c" = "$st" ] && [ "${#sum}" = 64 ]; then
    echo "sha256:$sum"
    return 0
  fi
  [ "$compute" = 1 ] || { echo "PENDING"; return 0; }
  sum=$(set -o pipefail; dd if="$f" bs=16M iflag=direct status=none 2>/dev/null | sha256sum | cut -c1-64) ||
    sum=$(sha256sum -- "$f" | cut -c1-64) || { echo "ERR cannot hash $f"; return 0; }
  st2=$(stat -c '%s %Y %i' -- "$f" 2>/dev/null)
  [ "$st2" = "$st" ] || { echo "ERR $f changed while hashing"; return 0; }
  { printf '%s %s\n' "$st" "$sum" > "$cache.$$" && mv -f -- "$cache.$$" "$cache"; } 2>/dev/null || rm -f -- "$cache.$$" 2>/dev/null
  echo "sha256:$sum"
}
