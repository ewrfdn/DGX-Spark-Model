#!/usr/bin/env bash
# stop_tp2.sh(在 head 上跑): 优雅停止两台上的服务容器（开启 Spill 时等待刷盘），并打印剩余容器/GPU 进程。
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
dsv41_load_config "$here"
export DSV41_CONFIG="$CFG"

"$here/serve_tp2.sh" stop
rc=$?
sleep 3
for t in $NODES; do
  ssh -n -o BatchMode=yes "$t" 'echo "$(hostname): containers: $(docker ps --format "{{.Names}}" | tr "\n" " ") gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"'
done
echo "STOP rc=$rc"
exit "$rc"
