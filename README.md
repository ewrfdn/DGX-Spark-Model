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
