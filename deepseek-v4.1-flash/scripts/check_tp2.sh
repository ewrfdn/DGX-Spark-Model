#!/usr/bin/env bash
# check_tp2.sh(在 head 上跑): 只做健康检查，不重启。
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
CFG=$(dsv41_resolve_config "$here")
export DSV41_CONFIG="$CFG"
"$here/serve_tp2.sh" check
