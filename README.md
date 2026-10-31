# DGX-Spark-Model

DGX Spark 集群上各模型的部署配置与教程仓库。

## 目录结构

```
<模型名>/
├── 部署教程-手把手.md    # 小白向完整部署教程,每步带命令
├── 部署流程.md           # 方案细节:原理、参数表、进阶选项、排障
└── config/               # 该模型的配置文件模板(如 local.sh.example)
```

## 已收录模型

| 模型 | 方案 | 硬件 | 状态 |
|---|---|---|---|
| [GLM-5.3-Flash](./glm-5.3-flash/) | [TensorFold EXL3 双机方案](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | 2x DGX Spark | 部署中 |
| [Qwen3.8-Flash-Next](./qwen3.8-flash-next/) | 单机 [TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) / 双机 [vLLM TP2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | 1x / 2x DGX Spark | ✅ 已跑通(单机 42–52 tok/s / 双机 11–15 tok/s) |
| [DeepSeek-V4.1-Flash](./deepseek-v4.1-flash/) | [TensorFold TP2](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks)(引擎 `deepseek-v41-zig`,Mia-AiLab EXL3 2.9bpw) | 2x DGX Spark | 已部署(`:8080`,1M 上下文 / 4 流,实测 C1 61 tok/s、C4 125 tok/s) |
