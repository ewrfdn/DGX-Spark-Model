#!/usr/bin/env bash
# build.sh: 兼容旧版宿主机挂载模式的编译脚本 (在 nvcr.io/nvidia/pytorch:26.07-py3 里编译 TensorFold 原生引擎)
# 推荐优先使用 ./scripts/build_image.sh 构建内置最新 0001..0068 补丁的独立推理镜像。
#
# P0 修正:
#   去掉原先写死的 `-Dtarget=aarch64-linux-gnu.2.28`（该参数会降级为 generic ARMv8-A，关闭 GB10 硬件 CRC32/SHA256）。
#   在 GB10 原生编译直接启用宿主 ARMv9.2 指令集；如需兼容旧 glibc 可传 ZIG_BUILD_EXTRA="-Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline+crc+sha2"。
set -euo pipefail
export PATH=/work/zig-aarch64-linux-0.17.0:$PATH
export ZIG_GLOBAL_CACHE_DIR=/work/zigcache
export HOME=/work/home
mkdir -p "$HOME" "$ZIG_GLOBAL_CACHE_DIR"
cd /work/engine
zig version
nvcc --version | tail -2
date -u
# shellcheck disable=SC2086
zig build install native \
  -Doptimize=ReleaseFast \
  -Dnvcc=/usr/local/cuda/bin/nvcc \
  -Dsm="${SM:-121}" \
  ${ZIG_BUILD_EXTRA:-} \
  -j"$(nproc)" 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "BUILD rc=$rc"
[ "$rc" = 0 ] || exit "$rc"
date -u
ls -la zig-out/bin zig-out/native/bin 2>&1
