#!/bin/bash
# build.sh: 在 nvcr.io/nvidia/pytorch:26.07-py3 里编译 TensorFold 原生引擎 (GB10, aarch64, sm_121)
set -e
export PATH=/work/zig-aarch64-linux-0.17.0:$PATH
export ZIG_GLOBAL_CACHE_DIR=/work/zigcache
export HOME=/work/home
mkdir -p "$HOME" "$ZIG_GLOBAL_CACHE_DIR"
cd /work/engine
zig version
nvcc --version | tail -2
date -u
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 2>&1 | tail -40
echo "BUILD rc=${PIPESTATUS[0]}"
date -u
ls -la zig-out/bin 2>&1 | head -20
