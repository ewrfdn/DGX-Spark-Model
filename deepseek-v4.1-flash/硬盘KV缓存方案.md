# 硬盘做 KV Cache(Spill Tier)方案

> 结论先行:**能**,而且有现成实现,不用自己写。
> 官方仓库 [MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks)
> 已经把 Spill v2 移植进同一个原生引擎,开关是 `SPILL_GIB`。
> 我们此前基于 tonyd2wild 的部署没有这套补丁,这是长上下文缓存失效的**真正原因**。

---

## 1. 问题回顾:为什么之前 250K 命中率是 0

实测证据(2026-10-31,new-api 消费日志的 `frt` 字段):

| prompt tokens | 首 token 时间 |
|---|---|
| 248,459 | 137.3 s |
| 251,979 | 141.1 s |
| 253,378 | 140.7 s |
| 253,575 | 140.6 s |
| 253,983 | 141.8 s |

长度几乎不变而耗时几乎不变,折算 prefill 速率稳定 **1.80K tok/s** —— 每次都在全量重算,
缓存贡献为零。对照我自己造的 168K 测试能命中(`resumed at 165888`),说明**不是缓存逻辑坏了**,
而是**大规模长对话的场景没被覆盖**。

**你的判断是对的:不是代码 bug,是这套部署缺少对应的引擎特性。**

---

## 2. Spill 是什么

引自官方 `docs/spill-and-kv-sharding.md`:

> **Spill v2** (`SPILL_GIB`): each kept prompt state is written, when it is kept, to each Spark's own
> local disk in 2,048-token blocks that later turns of the conversation share, and read back instead of
> a new prefill: **after eviction, after a restart, after a crash**. Room is made by conversation
> (old turns past the newest few, abandoned forks, whole conversations least recently used, shared
> prefixes last), with an optional per-conversation quota.

要点:

- 写的是**每台 Spark 自己的本地 NVMe**,不是共享盘,不是 NFS
- 按 **2048 token 块**切分,后续轮次**共享已写块**
- 用途覆盖:**被淘汰之后、重启之后、崩溃之后**都能读回
- 空间回收按**会话**维度做(旧轮次 → 废弃分支 → LRU 会话 → 共享前缀最后)

第二条(`after eviction / after a restart`)正是我们需要的:即使内存缓存被挤掉,硬盘上还在。

---

## 3. 磁盘占用(官方给出的计算值)

| 项目 | 大小 |
|---|---|
| 每 token(复制模式) | ~890 字节 |
| 每 2048 token 块 | 1,822,720 字节(≈1.74 MiB) |
| 100K token 对话 | ≈85 MiB |
| 1M 全窗口 | ≈890 MiB |
| 每个保留点额外 | 6.4 MiB |

启用 KV sharding 后,owner 那台写 ~890 字节/token,另一台只 ~170 字节/token。

我们的机器:head 3.7T(约 2.1T 空闲)、worker 3.1T 空闲,**空间完全不是问题**。
默认 `SPILL_GIB=64` 即可,还能用 `SPILL_QUOTA_GIB` 给单个会话限额。

---

## 4. 启用方式

官方 `scripts/local.sh.example` 给出:

```bash
SPILL_GIB=64                  # 上限,每台 Spark 64 GiB(0 = 关闭)
SPILL_DIR=...                 # rank 0 的 spill 目录(默认在 HF_CACHE 旁;绝不能在 NFS / 导出盘)
WORKER_SPILL_DIR=...          # rank 1 的目录(默认同 SPILL_DIR)
SPILL_FLOOR_GIB=50            # 始终保留的空闲磁盘
SPILL_QUOTA_GIB=16            # 单会话限额(0 = 不限)
SPILL_MIN_TOKENS=8192         # 短于此的状态不写
```

**重要约束**(官方明确写的,违反就拒绝启动):

- 目录**不能**在 NFS 上、不能是 tmpfs、不能在海量导出目录内
- 必须在 `SPILL_FLOOR_GIB` 之上还有足够空闲
- 两台机器都做检查,任一台不过就拒绝启动
- 目录属主必须是运行用户(不能是 root)
- 显式设置 `SPILL_GIB` 后,检查失败是**拒绝启动**而不是静默降级

另外一条我们**已经满足**的前置条件:

> The ranks' Engram folders must list the same files.

我们的 `engram/` 在 head 上已经是**实体文件**(17:12 拷贝完成),不是符号链接,正好符合
`HEAD_ENGRAM_DIR` 的要求。如果当初偷懒用软链接,这里会报 `FingerprintMismatch`。

---

## 5. 为什么不能只打补丁了事

我们现有的引擎来自 **tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks**,分支
`deepseek-v41-zig @ 3ff3b2a` + cherry-pick `5940208` + 手写补丁 0002/0003。

MiaAI 的补丁基线是 bertholomus/TensorFold 的 **`fbbc9713`**(2026-10-23 分支 tip),
两者**不是祖先关系**。

实测结果:

```
0013-core-spill-store.patch    ✓ 可应用
0014-core-cache-tier.patch     ✗ 冲突
0016-dsv41-spill-blocks.patch  ✗ 冲突
```

所以**不能**在我们当前树上零散打补丁。正确做法是**基于 `fbbc9713` 重建**:

```bash
git checkout fbbc9713
for f in /tmp/miaai/zig/patches/*.patch; do git apply "$f"; done
```

实测 **33 个补丁全部干净应用,0 失败**。

---

## 6. 实施步骤

### 6.1 引擎重建(已完成并验证)

```bash
cd /home/sakana/dgx-deepseek-tp2/engine
git branch -f backup-tp2-lane HEAD          # 先备份原分支
git checkout fbbc9713
for f in /tmp/miaai/zig/patches/*.patch; do git apply "$f" || echo "FAIL $f"; done
# 在容器里编译
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast \
    -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

产物在 `zig/zig-out/native/bin/tensorfold-native`,拷到 `engine-bin/` 并 rsync 到 worker。

### 6.2 准备 spill 目录

两台机器各自建目录(必须是自己有写权限的本地盘):

```bash
mkdir -p /home/sakana/dsv41-spill
```

### 6.3 配置

在 `config.tp2.env` 里加:

```bash
SPILL_GIB=64
SPILL_DIR=/home/sakana/dsv41-spill
WORKER_SPILL_DIR=/home/sakana/dsv41-spill
SPILL_FLOOR_GIB=50
SPILL_MIN_TOKENS=8192
```

然后把对应 flag 传进 `serve_tp2.sh` 的 rank 0 命令(`--spill-gib` 等),
rank 1 由 lanes 接收同名参数。

### 6.4 验证

1. `--spill-gib` 出现且服务能起
2. 长对话连续两轮:`resumed at` 不为 0
3. **重启服务后**同一个 prompt 仍能 `resumed`(这是 spill 的核心价值)
4. 检查 spill 目录里有文件生成、大小合理
5. new-api 侧 `frt` 从 ~140s 降到秒级

---

## 7. 回滚

```bash
cd engine && git checkout backup-tp2-lane
# 重新编译旧引擎,或直接还原备份二进制
```

spill 文件留在目录里不占运行内存,`rm -rf /home/sakana/dsv41-spill/*` 即可回收。

---

## 8. 附带收益:KV Sharding

同一套补丁还带来 `KV_SHARD=1`,官方说明:

> **The same `POOL` costs about half the KV memory a Spark** (owner-only planes): measured at
> 2,101,248 positions, **1,062 MiB a rank against 1,783 MiB replicated**, for 1-4% decode.

也就是 KV 内存**省一半**,代价是解码慢 1–4%。配合 spill 一起用。

注意配套要求:`KV_SHARD=1` 时 `POOL` 必须是 `CONTEXT` 整数倍的整窗口切片,
1M 窗口对应 `POOL=2101248`。填 `POOL=1048576` 会被拒绝(切不出两个整窗口)。

---

## 9. 风险与注意

| 风险 | 说明 | 应对 |
|---|---|---|
| 引擎基线变更 | 新补丁基于 `fbbc9713`,与我们原来的 `55a34467` 不同 | 已备份分支 `backup-tp2-lane`,可随时切回 |
| 补丁引入新行为 | 33 个补丁不只 spill,还含 KV sharding、pair-broken 等 | 全部保持默认关闭,只开 `SPILL_GIB` |
| `pair-broken` 默认开启 | 官方说这个"with two ranks whatever the settings"都生效 | 它只在失败后动作,正常请求不受影响;可用 `TF_DS_PAIR_BROKEN=degrade` 退出 |
| 目录检查严格 | 不合格直接拒绝启动 | 用本地盘、自己属主、避开 NFS/tmpfs |
| 磁盘写放大 | 每 token ~890 字节 | 64 GiB 上限 + 50 GiB 保留,足够 |

---

## 10. 参考

- 方案原文:`https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/main/docs/spill-and-kv-sharding.md`
- 补丁目录:`zig/patches/0013-core-spill-store.patch` ~ `0022-dsv41-kv-shard-ring.patch`
- 上游出处:Robert Wojciechowski (@rwojo) 的 GLM-5.3-Flash 移植,PR #78 / #124 / #128 / #152 / #149
- 完整流程:`docs/PORT-RECIPE.md`
