#!/bin/bash
# precheck_tp2.sh (在 head 上跑): 检查双机是否具备启动 TP2 lane 的条件。任何一项失败返回 1。不启动任何东西。
# 用法: DSV41_CONFIG=scripts/local.env ./precheck_tp2.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
CFG=${DSV41_CONFIG:-$here/local.env}
[ -r "$CFG" ] || { echo "没有配置文件 $CFG(先 cp config/tp2.env.example scripts/local.env)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CFG"
read -ra N <<< "$NODES"
rc=0
bad() { echo "FAIL: $*"; rc=1; }

for t in "${N[@]}"; do
  out=$(ssh -n -o BatchMode=yes "$t" "
    echo \"avail \$(awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo)\"
    echo \"containers \$(docker ps --format '{{.Names}}' | tr '\n' ' ')\"
    echo \"gpuprocs \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\"
    echo \"image \$(docker image inspect $IMAGE >/dev/null 2>&1 && echo yes || echo no)\"
    echo \"cache \$(ls $CACHE_DIR/rank-cache 2>/dev/null | tr '\n' ' ')\"
    echo \"token \$(ls $CACHE_DIR/dsv41_token_map.json 2>/dev/null)\"
    echo \"kit \$(ls $KIT_DIR/MANIFEST $KIT_DIR/vision/gate_bias_vl.safetensors $KIT_DIR/rope-plain-cos.f32 $KIT_DIR/vision/torch_fmha_sm120.cubin 2>/dev/null | wc -l)\"
    echo \"model \$(ls $MODEL_DIR/config.json $MODEL_DIR/tokenizer.json $MODEL_DIR/chat_template.jinja $MODEL_DIR/model.safetensors.index.json 2>/dev/null | wc -l)\"
    echo \"shards \$(ls $MODEL_DIR/model-*-of-*.safetensors 2>/dev/null | wc -l)\"
    echo \"engine \$(ls $ENGINE_DIR/tensorfold $ENGINE_DIR/tf-dsv41-lanes $ENGINE_DIR/tf-dsv41-rank-cache 2>/dev/null | wc -l)\"
    echo \"engram \$(ls $ENGRAM_DIR/model-00047-of-00048.safetensors $ENGRAM_DIR/model-00048-of-00048.safetensors 2>/dev/null | wc -l)\"
    echo \"gid3 \$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3 2>/dev/null)\"
    echo \"hcas \$(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' ')\"") || { bad "$t ssh 连不上"; continue; }
  echo "== $t"
  echo "$out" | sed 's/^/   /'
  echo "$out" | grep -q "^containers .*$LANE" && bad "$t 已经在跑 $LANE"
  # GPU 上有别的进程时只警告,真正的判据是 MemAvailable(下面),因为统一内存里其它进程占的也是同一块
  g=$(echo "$out" | awk '/^gpuprocs/ {print $2}')
  [ "$g" = 0 ] || echo "   WARN: $t 上有 $g 个 GPU 进程,确认它们不抢内存"
  echo "$out" | grep -q '^image yes' || bad "$t 没有镜像 $IMAGE"
  echo "$out" | grep -q '^cache rank[01]of2-' || bad "$t 没有 rank cache"
  echo "$out" | grep -q '^token ' || bad "$t 没有 dsv41_token_map.json"
  [ "$(echo "$out" | awk '/^kit/ {print $2}')" = 4 ] || bad "$t kit 不完整(要 MANIFEST、vision 两个文件、rope 表)"
  [ "$(echo "$out" | awk '/^model/ {print $2}')" = 4 ] || bad "$t 模型目录不完整"
  [ "$(echo "$out" | awk '/^shards/ {print $2}')" = 39 ] || bad "$t 权重分片数不是 39"
  [ "$(echo "$out" | awk '/^engine/ {print $2}')" = 3 ] || bad "$t 引擎二进制缺文件"
  [ "$(echo "$out" | awk '/^engram/ {print $2}')" = 2 ] || bad "$t 缺 Engram 表(shard 47/48)"
  a=$(echo "$out" | awk '/^avail/ {print $2}')
  awk -v a="$a" 'BEGIN{exit !(a+0 >= 108)}' || bad "$t MemAvailable ${a} GiB < 108 (每节点要放 ~99 GiB 权重 + KV pool)"
done

# worker 到 head 的 RDMA 链路(两台网口命名不对称,只按目标地址 ping,不写死 -I 接口名)
ssh -n -o BatchMode=yes "${N[1]}" "ping -c 1 -W 2 $MASTER_IP >/dev/null 2>&1" \
  && echo "rail(worker -> $MASTER_IP) ok" \
  || bad "worker ping 不通 head $MASTER_IP"

[ $rc = 0 ] && echo "PRECHECK OK" || echo "PRECHECK FAILED"
exit $rc
