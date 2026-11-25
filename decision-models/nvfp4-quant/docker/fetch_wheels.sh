#!/usr/bin/env bash
# 下载构建镜像所需的 wheel（只有 flashinfer 本体，约 7.6 MB）。
# 可选：MIRROR=https://mirrors.aliyun.com/pypi/simple ./fetch_wheels.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="${FLASHINFER_VERSION:-0.6.3}"
MIRROR="${MIRROR:-}"
mkdir -p "$HERE/wheels"
args=(download --no-deps --dest "$HERE/wheels" "flashinfer-python==${VERSION}")
if [ -n "$MIRROR" ]; then
  args+=(-i "$MIRROR" --trusted-host "$(echo "$MIRROR" | awk -F/ '{print $3}')")
fi
python3 -m pip "${args[@]}"
ls -l "$HERE/wheels"
