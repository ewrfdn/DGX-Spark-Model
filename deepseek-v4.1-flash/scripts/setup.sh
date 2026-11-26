#!/usr/bin/env bash
# setup.sh(在 head 上跑): 准备双机 TP2 环境。
#   ./setup.sh image        构建内置最新 0001..0068 补丁的推理镜像 (推荐 P0 方式)
#   ./setup.sh sync-image   构建并将推理镜像同步到 worker (docker save | docker load)
#   ./setup.sh build        兼容旧方式: 在基础容器内编译引擎到 $ENGINE_DIR (含 tensorfold-native)
#   ./setup.sh kit          生成 kit 的 RoPE 表和 vision bias (约 2 分钟)
#   ./setup.sh rankcache    生成两个 rank 的 rank cache 和 token map (约 5 分钟, 纯 CPU)
#   ./setup.sh sync         把引擎/kit/rank cache/模型同步到 worker
#   ./setup.sh all          按模式依次执行 (USE_IMAGE_ENGINE=1 时走 sync-image; 否则 build) -> kit -> rankcache -> sync
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
dsv41_load_config "$here"

read -ra _nodes <<< "$NODES"
WORKER="${_nodes[1]}"
step=${1:?image|sync-image|build|kit|rankcache|sync|all}

do_image() {
  "$here/build_image.sh"
}

do_sync_image() {
  "$here/build_image.sh" --sync
}

do_build() {
  echo "== 编译引擎(宿主机挂载模式，容器 $BASE_IMAGE，原生 GB10 CPU 启用硬件 CRC32/SHA256)"
  cp -f "$here/build.sh" "$BASE/build.sh"
  chmod +x "$BASE/build.sh"
  docker rm -f tf1017-build >/dev/null 2>&1 || true
  docker run --rm --name tf1017-build --network host --user "$(id -u):$(id -g)" \
    -e "SM=${SM:-121}" -e "ZIG_BUILD_EXTRA=${ZIG_BUILD_EXTRA:-}" \
    -v "$BASE:/work" "$BASE_IMAGE" bash /work/build.sh
  mkdir -p "$ENGINE_DIR"
  cp -f "$BASE"/engine/zig-out/bin/* "$ENGINE_DIR/"
  # 修复原先漏拷 tensorfold-native 的 bug (它产出在 zig-out/native/bin/)
  if [ -x "$BASE/engine/zig-out/native/bin/tensorfold-native" ]; then
    cp -f "$BASE/engine/zig-out/native/bin/tensorfold-native" "$ENGINE_DIR/"
  else
    echo "FAIL: 缺少 $BASE/engine/zig-out/native/bin/tensorfold-native" >&2
    exit 1
  fi
  ls -la "$ENGINE_DIR/tensorfold-native" "$ENGINE_DIR/tf-dsv41-lanes"
}

do_kit() {
  echo "== 生成 kit 的 RoPE 表"
  if [ "$USE_IMAGE_ENGINE" = 1 ] && docker run --rm --network none --entrypoint test "$IMAGE" -r /opt/tensorfold/tools/inputs/make_rope.py 2>/dev/null; then
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$MODEL_DIR:/model:ro" -v "$KIT_DIR:/kit" \
      --entrypoint python3 "$IMAGE" /opt/tensorfold/tools/inputs/make_rope.py /model /kit
  else
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$MODEL_DIR:/model:ro" -v "$KIT_DIR:/kit" -v "$BASE/tfpy/src:/tfpy:ro" \
      --entrypoint bash "$BASE_IMAGE" -c 'export PYTHONPATH=/tfpy; python3 /kit/tools/make_rope.py /model /kit'
  fi
  echo "== 生成 vision bias(从 EXL3 包内的 *.ffn.gate.bias_vl 张量)"
  ( cd "$MODEL_DIR" && python3 "$KIT_DIR/tools/make_bias_vl.py" . "$KIT_DIR/vision/gate_bias_vl.safetensors" )
  bash "$KIT_DIR/tools/verify_kit.sh" || true
}

do_rankcache() {
  echo "== 两个 rank 的 rank cache"
  for r in 0 1; do
    out="$BASE/cache"; [ "$r" = 1 ] && out="$BASE/cache-r1"
    mkdir -p "$out/rank-cache"
    if [ "$USE_IMAGE_ENGINE" = 1 ]; then
      docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$MODEL_DIR:/model:ro" -v "$out/rank-cache:/out" \
        --entrypoint tf-dsv41-rank-cache "$IMAGE" /model /out "$r" 2
    else
      nice -n 19 "$ENGINE_DIR/tf-dsv41-rank-cache" "$MODEL_DIR" "$out/rank-cache" "$r" 2
    fi
  done
  echo "== token map"
  if [ "$USE_IMAGE_ENGINE" = 1 ] && docker run --rm --network none --entrypoint test "$IMAGE" -r /opt/tensorfold/tools/inputs/make_token_map.py 2>/dev/null; then
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$MODEL_DIR:/model:ro" -v "$BASE/cache:/cache" \
      --entrypoint python3 "$IMAGE" /opt/tensorfold/tools/inputs/make_token_map.py /model /cache/dsv41_token_map.json
  else
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$MODEL_DIR:/model:ro" -v "$BASE/cache:/cache" -v "$BASE/engine/tools/dsv41-zig/inputs:/tools:ro" \
      --entrypoint python3 "$BASE_IMAGE" /tools/make_token_map.py /model /cache/dsv41_token_map.json
  fi
}

do_sync() {
  echo "== 同步到 worker: $WORKER"
  ssh -o BatchMode=yes "$WORKER" "mkdir -p $BASE/{model,engram,kit,cache/rank-cache,logs,engine-bin}"
  if [ "$USE_IMAGE_ENGINE" != 1 ]; then
    rsync -a --delete "$ENGINE_DIR/" "$WORKER:$ENGINE_DIR/"
  fi
  rsync -a --delete "$KIT_DIR/" "$WORKER:$KIT_DIR/"
  rsync -aL "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
  rsync -aL "$ENGRAM_DIR/" "$WORKER:$ENGRAM_DIR/"
  rsync -a "$BASE/cache-r1/rank-cache/" "$WORKER:$BASE/cache/rank-cache/"
  rsync -a "$BASE/cache/dsv41_token_map.json" "$WORKER:$BASE/cache/"
  echo "同步完成"
}

case $step in
  image) do_image ;;
  sync-image) do_sync_image ;;
  build) do_build ;;
  kit) do_kit ;;
  rankcache) do_rankcache ;;
  sync) do_sync ;;
  all)
    if [ "$USE_IMAGE_ENGINE" = 1 ]; then do_sync_image; else do_build; fi
    do_kit
    do_rankcache
    do_sync
    ;;
  *) echo "image|sync-image|build|kit|rankcache|sync|all" >&2; exit 2 ;;
esac
