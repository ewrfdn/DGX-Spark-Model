#!/usr/bin/env bash
# build_image.sh: 构建包含最新 MiaAI-Lab 补丁集 (0001..0068) 的 TensorFold DeepSeek-V4.1-Flash 推理镜像，
# 并可选同步到 worker 节点。
#
# 用法:
#   ./scripts/build_image.sh                  # 拉取/复用上游补丁 -> 计算 hash -> 本地 docker build
#   ./scripts/build_image.sh --sync           # 构建完成后 docker save | ssh worker docker load 并校验
#   ./scripts/build_image.sh --rebuild        # --no-cache 强制重建
#   ./scripts/build_image.sh --dry-run        # 仅打印版本 pin、镜像 tag 与构建参数，不执行 build
#
# 离线/弱网环境:
#   MIAAI_DIR=/path/to/MiaAI-Lab-repo         # 使用本地已 clone 的补丁仓库（不访问 GitHub）
#   TF_LOCAL_REPO=/path/to/TensorFold         # 使用本地已 clone 的 TensorFold 仓库（含 TF_REF 与 INPUTS_REF）
#   ZIG_TARBALL=/path/to/zig-aarch64-linux-0.17.0.tar.xz  # 使用本地 Zig 压缩包
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
dsv41_load_config "$here"

DO_SYNC=0
REBUILD=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --sync) DO_SYNC=1 ;;
    --rebuild) REBUILD=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_REPO="${TF_REPO:-https://github.com/bertholomus/TensorFold.git}"
TF_REF="${TF_REF:-fbbc97132cd004237d9887b8d3185896efaedc5c}"
INPUTS_REF="${INPUTS_REF:-59402086381a42c74d53053443224eec2790b4c5}"
MIAAI_REPO="${MIAAI_REPO:-https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git}"
MIAAI_REF="${MIAAI_REF:-c7f2b85100f0a7b099c92a4e21825e30d93c9c51}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
ZIG_MIRRORS="${ZIG_MIRRORS-https://zig.linus.dev/zig}"
SM="${SM:-121}"
ZIG_BUILD_EXTRA="${ZIG_BUILD_EXTRA:-}"
IMAGE_REPO="${IMAGE_REPO:-tensorfold-dsv41}"
IMAGE_TAG_PREFIX="${IMAGE_TAG_PREFIX:-exl3-tp2}"
DOCKERFILE="$root/docker/Dockerfile"

[ -r "$DOCKERFILE" ] || { echo "找不到 $DOCKERFILE" >&2; exit 1; }

# 1. 准备上游补丁目录 (优先使用本地 MIAAI_DIR；否则检出到 $BASE/miaai-patches)
patch_src="${MIAAI_DIR:-$BASE/miaai-upstream}"
if [ -n "${MIAAI_DIR:-}" ]; then
  [ -d "$patch_src/zig/patches" ] || { echo "MIAAI_DIR 缺少 zig/patches: $patch_src" >&2; exit 1; }
  echo "== 使用本地补丁仓库: $patch_src"
else
  mkdir -p "$BASE"
  if [ ! -d "$patch_src/.git" ]; then
    echo "== 克隆上游补丁仓库: $MIAAI_REPO @ ${MIAAI_REF:0:12}"
    git init -q "$patch_src"
    git -C "$patch_src" remote add origin "$MIAAI_REPO"
  fi
  cur=$(git -C "$patch_src" rev-parse HEAD 2>/dev/null || true)
  if [ "$cur" != "$MIAAI_REF" ]; then
    echo "== 拉取补丁提交 $MIAAI_REF"
    git -C "$patch_src" fetch -q --depth 1 origin "$MIAAI_REF"
    git -C "$patch_src" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$patch_src" rev-parse HEAD)" = "$MIAAI_REF" ] || { echo "补丁仓库 HEAD 不匹配 $MIAAI_REF" >&2; exit 1; }
fi

n_engine=$(find "$patch_src/zig/patches" -maxdepth 1 -name '*.patch' | wc -l | tr -d ' ')
n_inputs=$(find "$patch_src/zig/patches-inputs" -maxdepth 1 -name '*.patch' | wc -l | tr -d ' ')
[ "$n_engine" -ge 40 ] || { echo "补丁数量异常 ($n_engine < 40)，请检查 $patch_src/zig/patches" >&2; exit 1; }

# 校验关键 P0 补丁存在
for must in \
  0001-c7-indexer-split.patch \
  0008-pool-window.patch \
  0013-core-spill-store.patch \
  0015-dsv41-kv-shard-attn.patch \
  0016-dsv41-spill-blocks.patch \
  0019-dsv41-spill-kv-shard.patch \
  0033-graph-cap.patch \
  0040-restore-room.patch \
  0041-service-between-pieces.patch \
  0042-kv-shard-pieces.patch \
  0043-picture-resume-skip.patch \
  0045-cut-in-tail.patch \
  0059-lane-thread-binds-context.patch; do
  [ -r "$patch_src/zig/patches/$must" ] || { echo "缺少关键 P0 补丁: $must" >&2; exit 1; }
done

# 2. 计算确定性 PATCHES_HASH (与镜像标签、Spill v2 指纹绑定)
PATCHES_HASH=$(
  {
    LC_ALL=C; export LC_ALL
    for p in "$patch_src"/zig/patches/*.patch "$patch_src"/zig/patches-inputs/*.patch; do
      [ -e "$p" ] && cat "$p"
    done
    cat "$DOCKERFILE"
    echo "$TF_REPO $TF_REF $INPUTS_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE sm=$SM extra=$ZIG_BUILD_EXTRA"
  } | sha256sum | cut -c1-12
)

TARGET_IMAGE="${BUILD_IMAGE_TAG:-${IMAGE_REPO}:${IMAGE_TAG_PREFIX}-${PATCHES_HASH}}"

echo "== 构建参数汇总"
echo "   BASE_IMAGE     = $BASE_IMAGE"
echo "   TF_REF         = ${TF_REF:0:12} (inputs: ${INPUTS_REF:0:12})"
echo "   MIAAI_REF      = ${MIAAI_REF:0:12} ($n_engine engine + $n_inputs inputs patches)"
echo "   ZIG            = $ZIG_VERSION (native GB10 CPU -> hardware CRC32/SHA256)"
echo "   PATCHES_HASH   = $PATCHES_HASH"
echo "   TARGET_IMAGE   = $TARGET_IMAGE"
echo "   SYNC_TO_WORKER = $DO_SYNC"

if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] 校验通过，未执行 docker build"
  exit 0
fi

# 3. 组装干净的临时 docker build context
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$TARGET_IMAGE" 2>/dev/null || true)
if [ "$REBUILD" = 0 ] && [ "$built_hash" = "$PATCHES_HASH" ]; then
  echo "== 镜像 $TARGET_IMAGE 已就绪 (tf.patches=$built_hash)，跳过构建（传 --rebuild 强制重建）"
else
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
    echo "本机缺少基础镜像 $BASE_IMAGE，请先 docker pull 或从 worker 导入 (见 部署流程.md §3.1)" >&2
    exit 1
  }
  ctx=$(mktemp -d "${TMPDIR:-/tmp}/dsv41-img-ctx.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches" "$ctx/patches-inputs" "$ctx/offline"
  cp "$DOCKERFILE" "$ctx/Dockerfile"
  cp "$patch_src"/zig/patches/*.patch "$ctx/patches/"
  cp "$patch_src"/zig/patches-inputs/*.patch "$ctx/patches-inputs/"

  # 可选离线缓存，避免 docker build 内受宿主机失效代理/弱网影响
  if [ -n "${ZIG_TARBALL:-}" ] && [ -r "$ZIG_TARBALL" ]; then
    echo "== 注入离线 Zig 包: $ZIG_TARBALL"
    cp "$ZIG_TARBALL" "$ctx/offline/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"
  fi
  if [ -n "${TF_LOCAL_REPO:-}" ] && [ -d "$TF_LOCAL_REPO" ]; then
    echo "== 注入离线 TensorFold 仓库: $TF_LOCAL_REPO"
    git clone --bare "$TF_LOCAL_REPO" "$ctx/offline/tensorfold.git"
  fi

  build_args=(
    --network host
    -t "$TARGET_IMAGE"
    -t "${IMAGE_REPO}:${IMAGE_TAG_PREFIX}-latest"
    -f "$ctx/Dockerfile"
    --build-arg "BASE_IMAGE=$BASE_IMAGE"
    --build-arg "TF_REPO=$TF_REPO"
    --build-arg "TF_REF=$TF_REF"
    --build-arg "INPUTS_REF=$INPUTS_REF"
    --build-arg "ZIG_VERSION=$ZIG_VERSION"
    --build-arg "ZIG_SHA256=$ZIG_SHA256"
    --build-arg "ZIG_MIRRORS=$ZIG_MIRRORS"
    --build-arg "SM=$SM"
    --build-arg "ZIG_BUILD_EXTRA=$ZIG_BUILD_EXTRA"
    --build-arg "PATCHES_HASH=$PATCHES_HASH"
    --build-arg "MIAAI_REF=$MIAAI_REF"
  )
  [ "$REBUILD" = 1 ] && build_args+=(--no-cache)

  echo "== 开始构建 $TARGET_IMAGE (两轮 Zig 编译，GB10 上约 5–6 分钟，不需要 GPU)"
  docker build "${build_args[@]}" "$ctx"
  rm -rf -- "$ctx"
  trap - EXIT
fi

# 验证产物能力（无 GPU、无网络）
echo "== 校验镜像内二进制与 Spill 开关"
docker run --rm --network none --entrypoint bash "$TARGET_IMAGE" -c '
  command -v tensorfold-native tf-dsv41-lanes tf-dsv41-rank-cache >/dev/null
  tensorfold-native capabilities --json | grep -q -- "--spill-gib"
  echo "OK: $(command -v tensorfold-native)"
'

if [ "$DO_SYNC" = 1 ]; then
  read -ra _nodes <<< "$NODES"
  worker="${_nodes[1]}"
  head_id=$(docker image inspect -f '{{.Id}}' "$TARGET_IMAGE")
  worker_id=$(ssh -o BatchMode=yes "$worker" "docker image inspect -f '{{.Id}}' '$TARGET_IMAGE' 2>/dev/null || true")
  if [ "$worker_id" = "$head_id" ]; then
    echo "== worker ($worker) 已有相同镜像 $TARGET_IMAGE ($head_id)"
  else
    echo "== 同步镜像 $TARGET_IMAGE 到 $worker (docker save | docker load)"
    docker save "$TARGET_IMAGE" | ssh -o BatchMode=yes "$worker" "docker load && docker tag '$TARGET_IMAGE' '${IMAGE_REPO}:${IMAGE_TAG_PREFIX}-latest'"
    worker_id=$(ssh -o BatchMode=yes "$worker" "docker image inspect -f '{{.Id}}' '$TARGET_IMAGE'")
    [ "$worker_id" = "$head_id" ] || { echo "FAIL: worker 镜像 ID ($worker_id) != head ($head_id)" >&2; exit 1; }
    echo "== 同步完成，两台镜像 ID 一致: $head_id"
  fi
fi

echo "== 完成。请在配置文件中设置:"
echo "   USE_IMAGE_ENGINE=1"
echo "   IMAGE=$TARGET_IMAGE"
