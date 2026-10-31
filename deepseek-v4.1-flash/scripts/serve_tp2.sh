#!/bin/bash
# serve_tp2.sh start|stop|check|print  [uses DSV41_CONFIG]: the TP2 launch on two GB10 nodes. Adapted from the TP4 recipe's
# scripts/serve.sh (bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10, Apache-2.0) and the run commands of
# bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10 (Apache-2.0). Rank 1 (`tf-dsv41-lanes`) starts first, then rank 0
# (`tensorfold serve`); each runs in a container ($LANE) on its node via node_run.sh, fed over ssh.
set -u
here=$(cd "$(dirname "$0")" && pwd)
CFG=${DSV41_CONFIG:-$here/config.tp2.env}
[ -r "$CFG" ] || { echo "no $CFG" >&2; exit 1; }
source "$CFG"
read -ra N <<< "$NODES"
[ ${#N[@]} = 2 ] || { echo "NODES must name two nodes, rank 0 first" >&2; exit 1; }
U=http://127.0.0.1:$PORT
PRINT=0
SITE="IMAGE=$IMAGE ENGINE_DIR=$ENGINE_DIR KIT_DIR=$KIT_DIR MODEL_DIR=$MODEL_DIR ENGRAM_DIR=$ENGRAM_DIR CACHE_DIR=$CACHE_DIR LOG_DIR=$LOG_DIR SPILL_DIR=${SPILL_DIR:-}"
# 两台机器的 CX7 网卡/HCA 命名**不对称**(见部署流程 2.1),下面这些必须按节点分别给:
#   NET_IF / WORKER_NET_IF         各自互联网口名(NCCL_SOCKET_IFNAME)
#   RDMA_HCA / WORKER_RDMA_HCA    各自的 RDMA 设备名;worker 必须填,否则 NCCL 找不到 IB 设备会静默回退
#                                  socket 传输,随后在 ncclAllGather 报 "remote process exited"(6)。
# 实测对应关系(head -> worker):
#   rocep1s0f0(enp1s0f0np0)   <->  rocep1s0f1(enp1s0f1np1)
#   roceP2p1s0f0(enP2p1s0f0np0) <-> roceP2p1s0f1(enP2p1s0f1np1)

net_if_for() { case "$1" in "${N[0]}") echo "$NET_IF" ;; *) echo "${WORKER_NET_IF:-$NET_IF}" ;; esac; }
hca_for() { case "$1" in "${N[0]}") echo "$RDMA_HCA" ;; *) echo "${WORKER_RDMA_HCA:-$RDMA_HCA}" ;; esac; }
on() { ssh -o BatchMode=yes -n "$1" "$2"; }
rank() {   # rank NODE ENVS CMD
  local hca=$(hca_for "$1")
  # NCCL_SOCKET_IFNAME 在本机的 nvcr 容器里匹配不上 CX7 接口名:实测填接口名、IP、带 = 前缀都报
  # "Bootstrap : no socket interface found",只有 lo 或不填才正常。所以默认不传,让 NCCL 按路由表自己挑
  # (CX7 直连路由 metric 100 < rail1 的 101,会选中 rail2)。确需指定时用环境变量 NCCL_SOCKET_IF。
  local ifname="${NCCL_SOCKET_IF:-}"
  local link="NCCL_IB_HCA=$hca TF_RDMA_DEVICES=$hca${ifname:+ NCCL_SOCKET_IFNAME=$ifname}"
  if [ $PRINT = 1 ]; then echo "$1: env $2 $SITE $link node_run.sh $LANE '$3'"; return 0; fi
  ssh -o BatchMode=yes "$1" "env $2 $SITE $link bash -s -- $LANE '$3'" < "$here/node_run.sh" >/dev/null
}
check() {
  on "${N[0]}" "curl -sf -m 10 $U/v1/models | grep -q $SERVED_NAME && curl -sf -m 120 -H 'Content-Type: application/json' $U/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":4,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' | grep -q choices"
}
stop() { local h; for h in "${N[@]}"; do on "$h" "docker rm -f $LANE >/dev/null 2>&1; true"; done; }
start() {
  local fam envs up=0
  [ $PRINT = 1 ] || stop
  # rank 1 follows rank 0: its --pool equals rank 0's --context, and --streams equals rank 0's TF_DS_STREAMS (else the ranks wait on each other)
  # rank 1 的 --rdma 必须给 **worker 自己**的 HCA 名(两台命名差一位),不能照抄 head 的。
  # RDMA_RING=0 可以关掉引擎的 RDMA 环,让 TP 交换走 NCCL(与 NCCL_IB_DISABLE=1 搭配时用)。
  local ring=""; [ "${RDMA_RING:-1}" = 1 ] && ring="--rdma $(hca_for "${N[1]}")"
  fam="--engram /engram --token-map /cache/dsv41_token_map.json --pool $CONTEXT $ring --graphs 1 --side 1 --prefetch 1 --aio 1 --streams $STREAMS"
  rank "${N[1]}" "$ENGINE_ENV" "/engine/tf-dsv41-lanes /model /cache/rank-cache 1 2 $MASTER_IP $TP_PORT /kit $fam" || { echo "start rank 1 failed" >&2; return 1; }
  envs="TF_TP_WORLD=2 TF_TP_MASTER=$MASTER_IP TF_TP_PORT=$TP_PORT TF_DS_RANK_CACHE=/cache/rank-cache TF_DS_KIT=/kit"
  envs="$envs TENSORFOLD_CUDA_KERNELS=/kit/aot TF_DS_ENGRAM=/engram TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json"
  envs="$envs TF_DS_GRAPHS=1 TF_DS_HC_SIDE=1 TF_DS_L2_PREFETCH=1 TF_DS_ENGRAM_AIO=1 TF_DS_STREAMS=$STREAMS $ENGINE_ENV"
  rank "${N[0]}" "$envs" "/engine/tensorfold-native serve /model --host $HOST --port $PORT --name $SERVED_NAME --context $CONTEXT --parallel $PARALLEL $SERVER_FLAGS --backend cuda" || { echo "start rank 0 failed" >&2; return 1; }
  [ $PRINT = 1 ] && return 0
  for _ in $(seq 1 240); do
    sleep 5
    if on "${N[0]}" "curl -sf -m 5 $U/v1/models | grep -q $SERVED_NAME"; then up=1; break; fi
    on "${N[0]}" "docker ps --format '{{.Names}}' | grep -qx $LANE" || break
  done
  [ $up = 1 ] || { echo "rank 0 did not come up: see $LOG_DIR/$LANE.log on ${N[0]}" >&2; return 1; }
  check && echo "up and answering on ${N[0]}:$PORT" || { echo "up, but the chat check failed" >&2; return 1; }
}
case ${1:?start|stop|check|print} in
  start) start ;; stop) stop ;; check) check ;; print) PRINT=1; start ;;
  *) echo "start|stop|check|print" >&2; exit 2 ;;
esac
