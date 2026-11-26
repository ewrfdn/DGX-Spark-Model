# 硬盘做 KV Cache (Spill v2) + P0 长上下文 TTFT 优化方案

> 结论先行：**已落地完整的 P0 方案**（独立 Dockerfile 镜像构建 + 上游 `0001`–`0068` 全量补丁 + `KV_SHARD=1` + 自动扩 `POOL` + Spill v2 调优 + `TF_FILL_*` 分片预填充 + 优雅停服刷盘）。
>
> 此前基于 `tonyd2wild` (`3ff3b2a`) 的部署没有 Spill 补丁，且即便早期手动套过 33 补丁，也因为 **KV Pool 过小（仅 890 MiB）被迫频繁走盘、缺少 `0040`/`0059`/`0041`/`0042`/`0045` 补丁、编译降级关闭硬件 CRC32/SHA256、以及脚本未完整透传参数**，导致长上下文 TTFT 明显受影响。

---

## 1. 问题回顾：为什么 250K 命中率为 0、且仅开 Spill v2 时长上下文 TTFT 仍偏慢

1. **旧基线 (`3ff3b2a`) 命中率为 0**：实测 248K–254K prompt 首 token 耗时恒为 137–142 s（折算冷 prefill 1.80K tok/s），见 [部署流程.md §9.4](./部署流程.md#94--prompt-cache-对长上下文完全不生效引擎-bug2026-10-30-定位)。
2. **仅开 `SPILL_GIB` 而不做 P0 优化时的典型退化原因**：
   - **KV Pool 过小逼热对话上盘**：DeepSeek-V4.1-Flash 每 token 压缩 KV 仅 **890 B**（1M 全窗口仅 890 MiB）。不开 `KV_SHARD`、不扩 `POOL` 时，整个池子仅 1M positions，2~3 个长对话交替进来立刻互相驱逐，被迫频繁落盘/读盘。
   - **缺关键补丁导致静默回退冷 Prefill**：
     - 缺 `0040-restore-room`：并发流占池时恢复报 `ContextFull`，直接放弃读盘从 0 token 重算；
     - 缺 `0059-lane-thread-binds-context`：重启后首个磁盘恢复请求报 `CUDA_ERROR_INVALID_CONTEXT`，静默退回冷 Prefill；
     - 缺 `0041`/`0042`/`0045`：分片预填充（`TF_FILL_CHUNKS`）无法与 `KV_SHARD=1`/Spill 协作，且命中缓存的长对话不能按增量尾巴长度插队；
     - 缺 `0043-picture-resume-skip`：带图多轮恢复重复跑视觉塔（每轮 +5.5 s）。
   - **编译与参数拖慢读/写盘**：
     - 旧 `build.sh` 传 `-Dtarget=aarch64-linux-gnu.2.28` 未设 `-Dcpu`，降级为 generic ARMv8-A，关闭了 GB10（ARMv9.2）硬件 CRC32/SHA256；
     - 默认 `SPILL_STAGING_MIB=128` 扣掉小状态槽后读缓冲仅 26 槽（~45 MiB），且 `TF_SPILL_LOAD_MS=40` 切片偏碎；
     - 旧 `node_run.sh` 用 `bash -c "( $CMD ) | tee"` + `docker rm -f`，停服直接 SIGKILL 无法触发 Spill 优雅刷盘。

---

## 2. Spill v2 + KV Sharding 核心机制与空间账

| 项目 | 复制模式 (`KV_SHARD=0`) | 分片模式 (`KV_SHARD=1`, P0 默认) |
|---|---|---|
| 内存 KV 池 (`POOL=2101248`, 2×1M) | ~1,783 MiB / 卡 | **~1,062 MiB / 卡**（只比 1M 复制多 172 MiB，容量翻倍） |
| 磁盘占用（每 token / 每 2048 块） | 两台各 ~890 B / 1.74 MiB | owner ~890 B，**另一台仅 ~170 B（I/O 降 5.2×）** |
| 100K / 1M 对话单次磁盘块 | ~85 MiB / ~890 MiB | owner 同左，非 owner ~16 MiB / ~170 MiB |
| 每个保留点小状态 (`dsv41.small`) | 6.38 MiB (`6,684,672 B`) | 6.38 MiB |
| Decode 速度代价 | 基准 | 约 -1% ~ -4%（走 RDMA ring `0022`） |

> 两台机器本地 NVMe 空余均在 2–3 TiB 以上，`SPILL_GIB=64` + `SPILL_FLOOR_GIB=50` + `SPILL_QUOTA_GIB=16` 绰绰有余。
> 若 head 上同时跑其他服务导致剩余内存紧张，可将 `CONTEXT` 设为 `524288` 搭配 `POOL=2101248`（4 个完整 512K 窗口常驻内存，极少触发盘恢复）。

---

## 3. P0 已落地的具体改动清单

1. **标准化 Docker 推理镜像构建**：
   - 新增 [docker/Dockerfile](./docker/Dockerfile) 与 [scripts/build_image.sh](./scripts/build_image.sh)，固定上游 `bertholomus/TensorFold @ fbbc9713`、`INPUTS_REF @ 59402086` 与 `MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks @ c7f2b851`（完整 `0001`–`0068` 补丁）。
   - **原生 GB10 编译**（去掉 `-Dtarget=aarch64-linux-gnu.2.28` 降级），自动启用 ARMv9.2 硬件 CRC32/SHA256，加速 Spill 块校验与落盘。
   - 镜像自动打上 `tf.patches` / `tf.ref` / `tf.miaai` 标签，作为 Spill v2 兼容性指纹来源；支持 `MIAAI_DIR` / `TF_LOCAL_REPO` / `ZIG_TARBALL` 离线注入。
   - 同步修复了旧版宿主机编译路径（[scripts/build.sh](./scripts/build.sh) 去除了 CPU 降级，[scripts/setup.sh](./scripts/setup.sh) 补齐了 `zig-out/native/bin/tensorfold-native` 拷贝）。
2. **统一配置解析与启动/预检/优雅停服**：
   - 新增 [scripts/lib_config.sh](./scripts/lib_config.sh)（统一 `$DSV41_CONFIG` → `config.tp2.env` → `local.env`，含 `KV_SHARD` 最小池自动推导、Spill 参数与指纹校验）。
   - [scripts/serve_tp2.sh](./scripts/serve_tp2.sh) 完整透传 `KV_SHARD=1`、`POOL`/`TF_DS_POOL`/`TF_DS_WINDOW`、按节点独立的 `SPILL_DIR`/`WORKER_SPILL_DIR`、`SPILL_STAGING_MIB=256`、`TF_SPILL_LOAD_MS=120`、Spill 必需身份变量及按 rank 缓存的 `TF_DS_RANK_CACHE_SHA`、`TF_FILL_*`、`TF_DS_ENGRAM_LEAD=1`、`GRAPH_CAP=128`。
   - [scripts/node_run.sh](./scripts/node_run.sh) 改为 `exec > >(tee ...) 2>&1; exec $CMD`，确保推理进程成为容器 PID 1，`docker stop -t 90` 发送的 `SIGTERM` 直达引擎触发 Spill 刷盘，停服后自动安全交还 root 写出的 spill 文件属主。
   - [scripts/precheck_tp2.sh](./scripts/precheck_tp2.sh) 修复了原先写死 `rocep1s0f0` 的 bug，并前置校验镜像能力（`--spill-gib`）、两节点 Spill 目录本地文件系统/属主/剩余空间、以及两节点 `ENGRAM_DIR` 文件列表一致性（避免停服后再报 `FingerprintMismatch`）。
   - [scripts/start_tp2.sh](./scripts/start_tp2.sh) 修复了 `rc=$?` 被 `echo` 覆盖导致启动失败也返回 0 的静默 bug。

---

## 4. 实施与升级步骤

### 4.1 构建并同步 P0 推理镜像（head 执行，约 5–6 分钟，不需要 GPU）

```bash
cd <本仓库>/deepseek-v4.1-flash
cp config/tp2.env.example scripts/local.env   # 如已有 scripts/config.tp2.env，对照模板合入 P0 配置项
vi scripts/local.env                          # 确认 NODES / RDMA_HCA / WORKER_RDMA_HCA / SPILL_DIR

# 预览将构建的版本 pin 与镜像 hash
./scripts/build_image.sh --dry-run

# 构建并自动通过 docker save | ssh worker docker load 同步到 worker
./scripts/build_image.sh --sync
```

> 弱网/离线场景可提前准备好本地目录并通过环境变量注入：
> `MIAAI_DIR=/tmp/miaai TF_LOCAL_REPO=~/dgx-deepseek-tp2/engine ZIG_TARBALL=~/zig-aarch64-linux-0.17.0.tar.xz ./scripts/build_image.sh --sync`

### 4.2 准备本地 Spill 目录（两台各自本地 NVMe，属主必须为当前用户）

```bash
mkdir -p /home/sakana/dsv41-spill
ssh sakana@192.168.101.2 'mkdir -p /home/sakana/dsv41-spill'
```

### 4.3 预检、核对启动命令并重启服务

```bash
# 仅打印两台即将执行的完整 docker / env 命令（不改任何状态）
./scripts/serve_tp2.sh print

# 停旧服（注意在无在途长任务时操作）-> 预检 -> 启动
./scripts/stop_tp2.sh
./scripts/start_tp2.sh
```

---

## 5. 验证指标（确认 P0 生效）

1. **启动日志与能力**：
   - `logs/tp2-lane.log` 出现 `dsv41 spill: rank 0's store under /spill (64.0 GiB, staging 256 MiB); the ranks agreed` 及 `pool 2101248`。
2. **内存命中与磁盘恢复**：
   - 同会话长上下文二轮请求命中内存（`resumed at ...`，TTFT 降至亚秒~秒级）；
   - 执行 `./scripts/stop_tp2.sh`（确认优雅刷盘完成、`/home/sakana/dsv41-spill/native-spill-*/rank0/{blocks,points}` 有文件且属主已交还 `sakana`）后再 `./scripts/start_tp2.sh`，重发同一长 prompt，检查 `/metrics`：
     - `prompt_cache_disk_hits_committed` 增长；
     - `prompt_cache_restores_room_waits` / `spill_writes_cancelled_guard` / `spill_presence_refused` 保持极低或为 0；
     - 观察 `prompt_cache_restore_wait_ns`，评估是否需要进一步推进 P1 源码级优化（去除空闲 2 ms 轮询、双机并行 unpack、独立 CUDA copy stream + `-Dport_kernels`）。

---

## 6. 回滚

- **保留新镜像、仅关特性**：在 `scripts/local.env`（或 `config.tp2.env`）中设 `SPILL_GIB=0 KV_SHARD=0 POOL=""`，执行 `./scripts/stop_tp2.sh && ./scripts/start_tp2.sh`。
- **回退旧宿主机挂载模式**：设 `USE_IMAGE_ENGINE=0`、`IMAGE=nvcr.io/nvidia/pytorch:26.07-py3`、`SPILL_GIB=0`、`KV_SHARD=0`、`POOL=""`。
- **清理磁盘缓存**：停服后两台执行 `rm -rf /home/sakana/dsv41-spill/native-spill-*` 即可回收全部空间。

---

## 7. 参考

- 上游方案仓库：`https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`（当前固定提交 `c7f2b851`，补丁 `0001`–`0068`）
- Spill 与 KV Sharding 说明：`docs/spill-and-kv-sharding.md`、`docs/PORT-RECIPE.md`
- 补丁清单：`zig/patches/README.md`
