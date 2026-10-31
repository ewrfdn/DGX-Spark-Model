#!/bin/bash
# start_tp2.sh(在 head 上跑): 先 precheck,通过后启动双机 TP2 服务。
# 用法: ./start_tp2.sh            用 scripts/local.env
#       DSV41_CONFIG=/path/x.env ./start_tp2.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
export DSV41_CONFIG=${DSV41_CONFIG:-$here/local.env}
date -u
"$here/precheck_tp2.sh" || { echo "precheck 失败,不启动"; exit 1; }
"$here/serve_tp2.sh" start
echo "START rc=$?"
date -u
