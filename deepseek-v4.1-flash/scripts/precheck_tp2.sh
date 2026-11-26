#!/usr/bin/env bash
# precheck_tp2.sh (在 head 上跑): 检查双机是否具备启动 TP2 lane 的条件（含 P0 镜像/KV Sharding/Spill v2/Engram 一致性检查）。
# 任何一项失败返回 1，不停止也不启动任何容器。
# 用法: ./precheck_tp2.sh  或  DSV41_CONFIG=scripts/config.tp2.env ./precheck_tp2.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=lib_config.sh
source "$here/lib_config.sh"
dsv41_load_config "$here"

read -ra N <<< "$NODES"
rc=0
bad() { echo "FAIL: $*"; rc=1; }

echo "== 配置文件: $CFG"
dsv41_validate_settings || rc=1

eff_pool=$(dsv41_effective_pool)
echo "== 目标运行参数: USE_IMAGE_ENGINE=$USE_IMAGE_ENGINE IMAGE=$IMAGE CONTEXT=$CONTEXT POOL=$eff_pool KV_SHARD=$KV_SHARD SPILL_GIB=$SPILL_GIB STAGING=${SPILL_STAGING_MIB}MiB"

hca_for() { case "$1" in "${N[0]}") echo "$RDMA_HCA" ;; *) echo "${WORKER_RDMA_HCA:-$RDMA_HCA}" ;; esac; }
spill_dir_for() { case "$1" in "${N[0]}") echo "$SPILL_DIR" ;; *) echo "${WORKER_SPILL_DIR:-$SPILL_DIR}" ;; esac; }

engram_lists=()
spill_on=0
dsv41_spill_on && spill_on=1

for idx in 0 1; do
  t="${N[$idx]}"
  first_hca=$(hca_for "$t" | cut -d',' -f1)
  sdir=$(spill_dir_for "$t")
  out=$(ssh -n -o BatchMode=yes "$t" "
    echo \"avail \$(awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo)\"
    echo \"containers \$(docker ps --format '{{.Names}}' | tr '\n' ' ')\"
    echo \"gpuprocs \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\"
    echo \"image \$(docker image inspect '$IMAGE' >/dev/null 2>&1 && echo yes || echo no)\"
    echo \"cache \$(ls '$CACHE_DIR'/rank-cache 2>/dev/null | tr '\n' ' ')\"
    echo \"token \$(ls '$CACHE_DIR'/dsv41_token_map.json 2>/dev/null)\"
    echo \"kit \$(ls '$KIT_DIR'/MANIFEST '$KIT_DIR'/vision/gate_bias_vl.safetensors '$KIT_DIR'/rope-plain-cos.f32 '$KIT_DIR'/vision/torch_fmha_sm120.cubin 2>/dev/null | wc -l)\"
    echo \"model \$(ls '$MODEL_DIR'/config.json '$MODEL_DIR'/tokenizer.json '$MODEL_DIR'/chat_template.jinja '$MODEL_DIR'/model.safetensors.index.json 2>/dev/null | wc -l)\"
    echo \"shards \$(ls '$MODEL_DIR'/model-*-of-*.safetensors 2>/dev/null | wc -l)\"
    echo \"engram \$(ls '$ENGRAM_DIR'/model-00047-of-00048.safetensors '$ENGRAM_DIR'/model-00048-of-00048.safetensors 2>/dev/null | wc -l)\"
    echo \"gid3 \$(cat '/sys/class/infiniband/$first_hca/ports/1/gid_attrs/types/3' 2>/dev/null)\"
    echo \"hcas \$(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' ')\"
    if [ '$USE_IMAGE_ENGINE' = 1 ]; then
      if docker image inspect '$IMAGE' >/dev/null 2>&1; then
        echo \"engine_img \$(docker run --rm --network none --entrypoint bash '$IMAGE' -c 'command -v tensorfold-native >/dev/null && command -v tf-dsv41-lanes >/dev/null && echo ok || echo missing' 2>/dev/null)\"
        echo \"spill_cap \$(docker run --rm --network none --entrypoint tensorfold-native '$IMAGE' capabilities --json 2>/dev/null | grep -q -- '--spill-gib' && echo yes || echo no)\"
      else
        echo \"engine_img no_image\"
        echo \"spill_cap no_image\"
      fi
    else
      echo \"engine_host \$(ls '${ENGINE_DIR:-}/tensorfold-native' '${ENGINE_DIR:-}/tf-dsv41-lanes' '${ENGINE_DIR:-}/tf-dsv41-rank-cache' 2>/dev/null | wc -l)\"
    fi
    if [ -n '$sdir' ] && [ '$spill_on' = 1 ]; then
      p='$sdir'; [ -d \"\$p\" ] || p=\$(dirname \"\$p\")
      while [ ! -d \"\$p\" ] && [ \"\$p\" != / ]; do p=\$(dirname \"\$p\"); done
      fs=\$(findmnt -n -o FSTYPE -T \"\$p\" 2>/dev/null | head -1)
      src=\$(findmnt -n -o SOURCE -T \"\$p\" 2>/dev/null | head -1)
      avail_b=\$(df -B1 --output=avail \"\$p\" 2>/dev/null | tail -1 | tr -d ' ')
      uid=\$(stat -c '%u' \"\$p\" 2>/dev/null || echo -1)
      me=\$(id -u)
      echo \"spill_probe \$fs \$src \$avail_b \$uid \$me \$(readlink -m '$sdir')\"
    fi
  ") || { bad "$t ssh 连不上"; continue; }

  echo "== $t (rank $idx)"
  echo "$out" | sed 's/^/   /'
  echo "$out" | grep -q "^containers .*$LANE" && bad "$t 已经在跑 $LANE"
  g=$(echo "$out" | awk '/^gpuprocs/ {print $2}')
  [ "$g" = 0 ] || echo "   WARN: $t 上有 $g 个 GPU 进程，确认它们不抢内存"
  echo "$out" | grep -q '^image yes' || bad "$t 没有镜像 $IMAGE"
  echo "$out" | grep -q "^cache .*rank${idx}of2-" || bad "$t 缺少 rank${idx}of2-*.bin"
  echo "$out" | grep -q '^token ' || bad "$t 没有 dsv41_token_map.json"
  [ "$(echo "$out" | awk '/^kit/ {print $2}')" = 4 ] || bad "$t kit 不完整(要 MANIFEST、vision 两个文件、rope 表)"
  [ "$(echo "$out" | awk '/^model/ {print $2}')" = 4 ] || bad "$t 模型目录不完整"
  [ "$(echo "$out" | awk '/^shards/ {print $2}')" = 39 ] || bad "$t 权重分片数不是 39"
  [ "$(echo "$out" | awk '/^engram/ {print $2}')" = 2 ] || bad "$t 缺 Engram 表(shard 47/48)"

  if [ "$USE_IMAGE_ENGINE" = 1 ]; then
    echo "$out" | grep -q '^engine_img ok' || bad "$t 镜像 $IMAGE 缺少 tensorfold-native 或 tf-dsv41-lanes (请先 ./scripts/build_image.sh --sync)"
    if dsv41_spill_on; then
      echo "$out" | grep -q '^spill_cap yes' || bad "$t 镜像 $IMAGE 不支持 --spill-gib (请先重建 P0 镜像)"
    fi
  else
    [ "$(echo "$out" | awk '/^engine_host/ {print $2}')" = 3 ] || bad "$t 宿主机 $ENGINE_DIR 缺少 tensorfold-native / tf-dsv41-lanes / tf-dsv41-rank-cache"
  fi

  if dsv41_spill_on; then
    sp_line=$(echo "$out" | awk '/^spill_probe / {$1=""; sub(/^ /, ""); print}')
    if [ -z "$sp_line" ]; then
      bad "$t 无法探测 spill 目录 $sdir"
    else
      read -r sp_fs sp_src sp_avail sp_uid sp_me sp_real <<< "$sp_line"
      case "$sp_fs" in
        nfs|nfs4|cifs|smb|smb2|smb3|fuse.sshfs|sshfs|9p|ceph|glusterfs|lustre|tmpfs|ramfs|devtmpfs|overlay|overlayfs|squashfs|zram)
          bad "$t spill 目录 $sp_real 位于不支持的文件系统 ($sp_fs)，必须使用本地 NVMe" ;;
      esac
      case "$sp_src" in
        /dev/zram*) bad "$t spill 目录 $sp_real 位于 zram ($sp_src)，必须使用本地 NVMe" ;;
      esac
      [ "$sp_real" != / ] || bad "$t spill 目录不能是 /"
      if [ "$sp_uid" = 0 ] && [ "$sp_me" != 0 ]; then
        bad "$t spill 目录 $sp_real 属主为 root，请改为当前用户: sudo chown $sp_me $sp_real"
      fi
      floor_b=$(awk -v f="$SPILL_FLOOR_GIB" 'BEGIN {printf "%.0f", f * 1073741824}')
      awk -v a="${sp_avail:-0}" -v f="$floor_b" 'BEGIN {exit !(a+0 >= f+0)}' || \
        bad "$t spill 目录 $sp_real 剩余空间低于 SPILL_FLOOR_GIB ($SPILL_FLOOR_GIB GiB)"
      want_b=$(awk -v g="$SPILL_GIB" -v f="$SPILL_FLOOR_GIB" 'BEGIN {printf "%.0f", (g + f) * 1073741824}')
      awk -v a="${sp_avail:-0}" -v w="$want_b" 'BEGIN {exit !(a+0 < w+0)}' && \
        echo "   WARN: $t spill 目录 $sp_real 剩余空间小于 SPILL_GIB+SPILL_FLOOR_GIB ($SPILL_GIB + $SPILL_FLOOR_GIB GiB)"
    fi
  fi

  a=$(echo "$out" | awk '/^avail/ {print $2}')
  awk -v a="$a" 'BEGIN{exit !(a+0 >= 108)}' || bad "$t MemAvailable ${a} GiB < 108 (每节点需放 ~99 GiB 权重 + KV pool)"

  # 记录 Engram 文件列表用于两机指纹一致性检查 (Spill v2 强校验)
  if dsv41_spill_on; then
    elist=$(ssh -n -o BatchMode=yes "$t" "
      shopt -s dotglob nullglob
      for f in '$ENGRAM_DIR'/*; do
        [ -d \"\$f\" ] && [ ! -L \"\$f\" ] && continue
        printf '%s:%s\n' \"\${f##*/}\" \"\$(stat -L -c %s \"\$f\" 2>/dev/null)\"
      done | LC_ALL=C sort
    " 2>/dev/null || echo "__ERR__")
    engram_lists+=("$elist")
  fi
done

# Spill v2 启用时，校验两台机器 Engram 目录文件列表一致，避免停服启动后报 FingerprintMismatch
if dsv41_spill_on && [ "${#engram_lists[@]}" = 2 ]; then
  if [ "${engram_lists[0]}" != "${engram_lists[1]}" ]; then
    bad "两台机器的 ENGRAM_DIR 文件列表不一致，Spill v2 会报 FingerprintMismatch"
    echo "   head:   $(echo "${engram_lists[0]}" | tr '\n' ' ')"
    echo "   worker: $(echo "${engram_lists[1]}" | tr '\n' ' ')"
  else
    echo "engram fingerprint listing match: ok"
  fi
fi

# worker 到 head 的 RDMA 链路
ssh -n -o BatchMode=yes "${N[1]}" "ping -c 1 -W 2 $MASTER_IP >/dev/null 2>&1" \
  && echo "rail(worker -> $MASTER_IP) ok" \
  || bad "worker ping 不通 head $MASTER_IP"

[ "$rc" = 0 ] && echo "PRECHECK OK" || echo "PRECHECK FAILED"
exit "$rc"
