#!/bin/bash
# check_tp2.sh(在 head 上跑): 只做健康检查,不重启。
set -u
here=$(cd "$(dirname "$0")" && pwd)
export DSV41_CONFIG=${DSV41_CONFIG:-$here/local.env}
"$here/serve_tp2.sh" check
