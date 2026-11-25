# 用几个镜像跑这五个模型

**结论：1 个镜像就够，跑 5 个容器**（每个模型一个进程）。按 runtime 拆是 3 个镜像，但没必要——三者依赖完全一致
（torch 2.14.0 / transformers 5.17.0 / Python 3.12），拆开只是多维护两份 Dockerfile。

权重**不进镜像**，全部靠 volume 挂载；镜像里只放"推理代码"。

## 1. 镜像 / 容器 / 显存

| 容器 | 用的镜像 | 模型 | 权重(bf16) | 建议显存 | 端口 |
|---|---|---|---|---|---|
| `gw-bison` | `decision-gateway` | bison-1 (31B) | ~63 GB | 80 GB（H100/A100） | 8011 |
| `gw-decider` | 同上 | pplx-decider-v1.3 (26B) | ~52 GB | 80 GB | 8013 |
| `gw-elephant` | 同上 | elephant-1 (9B) | ~19 GB | 24 GB（4090 可跑） | 8012 |
| `gw-flash` | 同上 | pplx-decider-flash (4.5B) | ~9 GB | 16 GB | 8014 |
| `gw-glide` | 同上 | GLiDE-0.4B | ~0.8 GB | 2 GB（CPU 也能跑） | 8015 |

单卡 80G 只能同时驻留一个 26B/31B；小三个（9B + 4.5B + 0.4B = 29GB）可以挤在一张 48G 卡上。

## 2. 镜像里有什么、挂在外面的是什么

镜像 = 解释器 + 三套 runtime + 网关；模型目录 = 权重 + 该模型自带的推理代码。

| 层 | 内容 | 来源 |
|---|---|---|
| **镜像 `decision-gateway`** | `python:3.12-slim`、torch 2.14.0、torchvision 0.29.0、transformers 5.17.0、tokenizers 0.23.2、safetensors 0.8.0、fastapi/uvicorn、**qev**（OneJev `81ce62f` + `serve.patch`）、`app/` 网关 | 构建 |
| **挂载 `/models/bison-1`** | `config.json`、`readout.json`、`model-*.safetensors`、`model.safetensors.index.json`、`chat_template.jinja`、`gemma_image.json`、`processor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`generation_config.json` | 下载 |
| **挂载 `/models/elephant-1`** | 同上，外加该模型自带 runtime 文件：`engine.py`、`serve.patch`、`setup.sh`、`constraints.txt` | 下载 |
| **挂载 `/models/pplx-decider-v1.3`** | `config.json`、`decision_config.json`、`joint_config.json`、`readout.safetensors`、`model-*.safetensors`、**`source/`（`pplx_decider` 包）**、`locate-span/`、`chat_template.jinja`、`processor_config.json`、`tokenizer*`、`release-manifest.json`、`requirements*.txt` | 下载 |
| **挂载 `/models/pplx-decider-flash-v1.3`** | 同上 | 下载 |
| **挂载 `/models/GLiDE-0.4B`** | `config.json`、`decision_config.json`、`readout.safetensors`、`model-00001-of-00001.safetensors`、**`glide/`（推理包）**、`render_tokenizer/`、`tokenizer*`、`licenses/`、`special_tokens_map.json` | 下载 |

关键差别：

- **Bison / Elephant**：runtime **在镜像里**（qev + patch），模型目录只需要 config + readout 配置 + 权重。
- **Decider / GLiDE**：runtime **在模型目录里**（`source/`、`glide/`），镜像里没有；所以这两个目录**不能只拷权重**，
  必须整目录挂载，否则 `import pplx_decider` / `import glide` 会失败。
- 三个模型族都依赖 `readout.json` / `readout.safetensors`，缺了会退化成普通 LM 或直接报错。

## 3. 为什么不用一个容器跑五个

显存装不下（合计 ~145GB）。网关支持 `MAX_LOADED=1` 按需换入换出，但换入一个 63GB 的 Bison 要读盘几分钟，
在网关后面会超时。所以**生产上按模型拆容器**，`MAX_LOADED` 只在单模型调试时有意义。

## 4. 接 New API

每个容器都暴露 OpenAI 兼容的 `POST /v1/chat/completions`（网关里已实现），New API 当普通 OpenAI 渠道加即可：

| New API 字段 | 值 |
|---|---|
| 类型 | OpenAI |
| 代理地址 | `http://<host>:8012`（elephant 容器，以此类推） |
| 密钥 | 任意非空字符串 |
| 模型 | `elephant-1`（与容器 `MODEL` 同名即可） |

请求形状（New API 原样转发）：

```json
{
  "model": "elephant-1",
  "messages": [{"role": "user", "content": "{\"state\": {\"ticket\": \"...\"}, \"questions\": {\"tier\": {\"type\": \"choice\", \"criteria\": {\"small\": \"...\", \"mid\": \"...\"}}}}"}]
}
```

返回是标准 `chat.completion`，`choices[0].message.content` 里是 `answers` 的 JSON 字符串；也支持 `stream: true`（单条 SSE）。
也可以把 `state` / `questions` 直接放在请求顶层（New API 不会剥掉未知字段）。

不带 `questions` 时用内置的路由预设（`tier` + `escalate`），所以 New API 的"测试连接"能直接通过。

## 5. 启动

```bash
cd serving
docker build -t decision-gateway .          # 一次

# 只跑一个模型：改 MODEL 即可复用同一个镜像
docker run -d --gpus '"device=0"' -p 8012:8000 \
  -e MODEL=elephant-1 -e MAX_LOADED=1 \
  -v /home/sakana/models/decision-models:/models:ro \
  decision-gateway

# 或者用 compose（5 个服务，按 GPU 分配 device_ids）
docker compose up -d gw-glide gw-flash gw-elephant
```

> 注意：本机连不上 Docker Hub，镜像需要在能拉 `python:3.12-slim` 的机器上构建；构建产物约 6–7 GB。
