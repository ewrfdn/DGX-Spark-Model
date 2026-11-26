#!/usr/bin/env bash
# start_tp2.sh(在 head 上跑): 先 precheck，通过后启动双机 TP2 服务。
# 用法: ./start_tp2.sh
#       DSV41_CONFIG=/path/x.env ./start_tp2.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
CFG=$(dsv41_resolve_config "$here")
export DSV41_CONFIG="$CFG"

date -u
"$here/precheck_tp2.sh" || { echo "precheck 失败，不启动" >&2; exit 1; }
"$here/serve_tp2.sh" start
rc=$?
echo "START rc=$rc"
date -u
exit "$rc"
