# DeepSeek-V4.1-Flash on 2× DGX Spark(TensorFold TP2)

DeepSeek-V4.1-Flash-1017 用 [TensorFold](https://github.com/ashhart/TensorFold) 原生引擎在两台 DGX Spark 上做 TP2,
1M 上下文、4 条并发流,服务在 `:8080`(8000 被本机 spark-management 占用)。

- 方案来源:[tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks) · TP2 lane
- 引擎:`bertholomus/TensorFold` 分支 `deepseek-v41-zig` @ `3ff3b2a` + 3 个补丁
- recipe:`bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10` v0.6.1
- 权重:**Mia-AiLab EXL3 2.9bpw(EAQ v1)** —— 与原方案的 LibertAI 包不同,见 [部署流程.md §1](./部署流程.md#1-权重选择为什么用-mia-ailab-而不是文档默认的-libertai)
- Engram 表:base checkpoint 的 shard 47/48

## 目录

```
deepseek-v4.1-flash/
├── README.md              # 本文件
├── 部署教程-手把手.md      # 零基础,每步带命令
├── 部署流程.md             # 原理、参数表、踩坑记录、实测数字、排障
├── config/
│   └── tp2.env.example    # 配置模板 → 复制到 scripts/local.env
└── scripts/
    ├── local.env.example  # 同上(实际使用的文件名)
    ├── setup.sh           # 从零准备:build / kit / rankcache / sync
    ├── precheck_tp2.sh    # 启动前检查两台是否就绪
    ├── start_tp2.sh       # precheck + 启动
    ├── stop_tp2.sh        # 两台 docker rm -f
    ├── check_tp2.sh       # 只做健康检查
    ├── serve_tp2.sh       # 启动器本体(start|stop|check|print)
    └── node_run.sh        # 节点侧:在容器里跑引擎,日志进 logs/
```

## 快速开始

```bash
cp config/tp2.env.example scripts/local.env
vi scripts/local.env                 # 改 NODES / MASTER_IP / NET_IF / RDMA_HCA
./scripts/setup.sh all               # 只在第一次:编译、生成 kit、rank cache、同步到 worker
./scripts/precheck_tp2.sh            # 全 PASS
./scripts/start_tp2.sh
curl -s http://127.0.0.1:8080/v1/models
```

## 本集群实测环境

| | head(rank 0) | worker(rank 1) |
|---|---|---|
| 主机 | `spark-adeb` | `spark-d705` |
| CX7 rail 2 | `enP2p1s0f0np0` `192.168.101.1` | `enp1s0f1np1` `192.168.101.2` |
| CX7 rail 1 | `enp1s0f0np0` `192.168.100.1` | `enP2p1s0f1np1` `192.168.100.2` |
| RDMA | `rocep1s0f0`, `roceP2p1s0f0` | `rocep1s0f1`, `roceP2p1s0f1` |
| 部署目录 | `~/dgx-deepseek-tp2` | `~/dgx-deepseek-tp2` |

## 状态

本集群已**部署完成并在跑**,以下全部为实测:

- [x] Docker 镜像两台就位(`nvcr.io/nvidia/pytorch:26.07-py3`,id `15b8de8baabe`)
- [x] 引擎编译通过(14 个二进制,含 `tensorfold-native`)
- [x] 模型目录 + Engram 表(两台都是实体文件)
- [x] kit 生成(RoPE 4 个文件 sha256 全通过;vision bias 见 [§5.2](./部署流程.md#52-vision-bias-的-sha256-差异已知且无害))
- [x] rank 0 / rank 1 cache 与 token map
- [x] 同步到 worker
- [x] 权重完整性逐分片校验(39 个模型分片 + 2 个 Engram 分片)
- [x] RoCE 链路双向连通,GID index 3,path_mtu 5
- [x] 双机启动验收:`up and answering on 192.168.101.1:8080`

### 实测数字(本集群,不是方案宣称值)

| 项目 | 实测 |
|---|---|
| 服务地址 | `http://192.168.101.1:8080/v1`(不是 8000,见下节) |
| 模型名 | `deepseek-v4.1-flash`(另注册别名 `glm-5.3-flash`) |
| 权重加载 | rank 1 16.3 s |
| TTFT(短提示) | 0.12 – 0.14 s |
| 单路(C1) | 55 – 64 tok/s |
| 2 路并发(C2) | 90 tok/s 聚合(每路 45) |
| 4 路并发(C4) | 115 – 124 tok/s 聚合(每路 29 – 31) |
| 双机 RoCE 裸链路 | 97.5 / 105.5 Gbps 双向,延迟 1.42 µs |
| 上下文 / 并发流 | 1048576 / 4 |
| 常驻内存 | head 16.5 GiB,worker 1.5 GiB(其余权重页在统一内存池里) |

> 原方案宣称 1 路 73.1 / 2 路 106.0 / 4 路 135.8 tok/s;本集群单路低约 10%,4 路基本持平(124.8 vs 135.8)。
> 权重包不同(Mia-AiLab vs LibertAI)是主要原因。

## 端口说明

**本机 8000 已被 `spark-management`(uvicorn,PID 2915)长期占用**,所以本部署用 **8080**。
原方案的 `:8000` 假设该端口空闲;你的机器上 8000 空着的话,把 `scripts/local.env` 的 `PORT` 改回 8000 即可。

## 两台机器的命名不对称(最容易踩的坑)

两台 Spark 的 CX7 网口名和 RDMA 设备名**不是一一对应**。共用一份配置时,worker 找不到本机 HCA 会**静默退回 socket 传输**,
然后在 `ncclAllGather` 报 `remote process exited or there was a network error (6)`:

| | head(rank 0) | worker(rank 1) |
|---|---|---|
| rail 2 网口 | `enP2p1s0f0np0` | `enp1s0f1np1` |
| rail 2 RDMA | `rocep1s0f0` | **`rocep1s0f1`** |
| rail 1 RDMA | `roceP2p1s0f0` | **`roceP2p1s0f1`** |

`serve_tp2.sh` 已按节点分别取 `RDMA_HCA` / `WORKER_RDMA_HCA`。详见 [部署流程.md §2.1](./部署流程.md#21-两台的网口与-rdma-设备名不对称)。
