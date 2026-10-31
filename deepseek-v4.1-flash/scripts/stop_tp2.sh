#!/bin/bash
# stop_tp2.sh(在 head 上跑): 停掉两台上的服务容器,并打印剩余容器/GPU 进程。
set -u
here=$(cd "$(dirname "$0")" && pwd)
export DSV41_CONFIG=${DSV41_CONFIG:-$here/local.env}
# shellcheck disable=SC1090
source "$DSV41_CONFIG"
"$here/serve_tp2.sh" stop
sleep 3
for t in $NODES; do
  ssh -n -o BatchMode=yes "$t" 'echo "$(hostname): containers: $(docker ps --format "{{.Names}}" | tr "\n" " ") gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"'
done
echo "STOP rc=$?"
