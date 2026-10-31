#!/bin/bash
# setup.sh(在 head 上跑一次): 从零准备双机 TP2 环境的 4 个阶段。
#   ./setup.sh all          依次做 build-kit / rankcache / sync
#   ./setup.sh build        容器里编译引擎(约 3 分钟,不需要 GPU)
#   ./setup.sh kit          生成 kit 的 RoPE 表和 vision bias(约 2 分钟)
#   ./setup.sh rankcache    生成两个 rank 的 rank cache 和 token map(约 5 分钟,纯 CPU)
#   ./setup.sh sync         把引擎/kit/rank cache/模型同步到 worker
set -eu
here=$(cd "$(dirname "$0")" && pwd)
CFG=${DSV41_CONFIG:-$here/local.env}
[ -r "$CFG" ] || { echo "没有 $CFG" >&2; exit 1; }
source "$CFG"
WORKER=$(awk '{print $2}' <<< "$NODES")
step=${1:?build|kit|rankcache|sync|all}

do_build() {
  echo "== 编译引擎(容器 $IMAGE,无 GPU)"
  cat > "$BASE/build.sh" <<'INNER'
#!/bin/bash
set -e
export PATH=/work/zig-aarch64-linux-0.17.0:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zigcache HOME=/work/home
mkdir -p "$HOME" "$ZIG_GLOBAL_CACHE_DIR"
cd /work/engine
zig version; nvcc --version | tail -1
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 2>&1 | tail -30
ls -la zig-out/bin
INNER
  chmod +x "$BASE/build.sh"
  docker rm -f tf1017-build >/dev/null 2>&1 || true
  docker run --rm --name tf1017-build --network host --user "$(id -u):$(id -g)" \
    -v "$BASE:/work" "$IMAGE" bash /work/build.sh
  mkdir -p "$ENGINE_DIR"; cp -f "$BASE"/engine/zig-out/bin/* "$ENGINE_DIR/"
}

do_kit() {
  echo "== 生成 kit 的 RoPE 表"
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$MODEL_DIR:/model:ro" -v "$KIT_DIR:/kit" -v "$BASE/tfpy/src:/tfpy:ro" \
    --entrypoint bash "$IMAGE" -c 'export PYTHONPATH=/tfpy; python3 /kit/tools/make_rope.py /model /kit'
  echo "== 生成 vision bias(从 EXL3 包内的 *.ffn.gate.bias_vl 张量)"
  ( cd "$MODEL_DIR" && python3 "$KIT_DIR/tools/make_bias_vl.py" . "$KIT_DIR/vision/gate_bias_vl.safetensors" )
  bash "$KIT_DIR/tools/verify_kit.sh" || true
}

do_rankcache() {
  echo "== 两个 rank 的 rank cache"
  for r in 0 1; do
    out="$BASE/cache"; [ "$r" = 1 ] && out="$BASE/cache-r1"
    mkdir -p "$out/rank-cache"
    nice -n 19 "$ENGINE_DIR/tf-dsv41-rank-cache" "$MODEL_DIR" "$out/rank-cache" "$r" 2
  done
  echo "== token map"
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$MODEL_DIR:/model:ro" -v "$BASE/cache:/cache" -v "$BASE/engine/tools/dsv41-zig/inputs:/tools:ro" \
    --entrypoint python3 "$IMAGE" /tools/make_token_map.py /model /cache/dsv41_token_map.json
}

do_sync() {
  echo "== 同步到 worker: $WORKER"
  ssh -o BatchMode=yes "$WORKER" "mkdir -p $BASE/{model,engram,kit,cache/rank-cache,logs,engine-bin}"
  rsync -a --delete "$ENGINE_DIR/" "$WORKER:$ENGINE_DIR/"
  rsync -a --delete "$KIT_DIR/" "$WORKER:$KIT_DIR/"
  rsync -aL "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
  rsync -aL "$ENGRAM_DIR/" "$WORKER:$ENGRAM_DIR/"
  rsync -a "$BASE/cache-r1/rank-cache/" "$WORKER:$BASE/cache/rank-cache/"
  rsync -a "$BASE/cache/dsv41_token_map.json" "$WORKER:$BASE/cache/"
  echo "同步完成"
}

case $step in
  build) do_build ;; kit) do_kit ;; rankcache) do_rankcache ;; sync) do_sync ;;
  all) do_build; do_kit; do_rankcache; do_sync ;;
  *) echo "build|kit|rankcache|sync|all" >&2; exit 2 ;;
esac
