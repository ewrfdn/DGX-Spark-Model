# 一个容器，五个决策模型，一套 System One API

这五个模型**没有一个公共推理引擎**能原生加载：它们的"决策头"各不相同，每个厂商都自带 runtime。所以这里的做法是
**一个镜像里装三套 runtime + 一个 FastAPI 网关**，对外只暴露 `POST /v1/systemone`（TypeSafe Jev 协议），按请求里的
`model` 字段分发到对应 runtime。

## 模型 ↔ runtime 对照

| 目录（MODEL_ROOT 下） | 模型 | 架构 | runtime | 证据 |
|---|---|---|---|---|
| `bison-1` | Base Labs Bison 31B | `Gemma4ForConditionalGeneration` + `readout.json` (slot_lm, 256 slots) | **qev**（OneJev @81ce62f + `serve.patch`） | 仓库含 `readout.json`；elephant 的 `setup.sh` 固定该 commit |
| `elephant-1` | Base Labs Elephant 9B | `Qwen3_5ForConditionalGeneration` + `readout.json` | **qev** 同上 | 仓库含 `engine.py` / `serve.patch` / `setup.sh` |
| `pplx-decider-v1.3` | Perplexity Decider v1.3 (26B) | `Qwen3_5Model`（无 LM head）+ 25 个答案码 | **pplx_decider** | `source/pyproject.toml` → `pplx-decider 0.5.0`，torch 2.14 / transformers 5.17 |
| `pplx-decider-flash-v1.3` | Perplexity Decider Flash v1.3 (4.5B) | 同上 | **pplx_decider** | 同上 |
| `GLiDE-0.4B` | Fastino GLiDE 0.4B | 0.39B encoder + 255-way readout | **glide** | 仓库内含 `glide/` 包和 `glide/server.py` |

网关按目录特征自动识别 runtime，不需要改配置：

```python
if (path / "glide").is_dir():                 -> GlideAdapter
elif (path / "source/src/pplx_decider").is_dir(): -> PplxAdapter
elif (path / "readout.json").exists():        -> QevAdapter
```

## 目录

```
serving/
├── Dockerfile            # python:3.12-slim + torch 2.14 + transformers 5.17 + qev(打补丁)
├── serve.patch           # Base Labs 的 qev readout 补丁（新增 qev/readout.py）
├── entrypoint.sh         # 把各模型自带的 source/src、glide 加进 PYTHONPATH
├── docker-compose.yml
└── app/main.py           # 网关：/v1/systemone、/v1/decisions、/v1/models、/health
```

## 构建与启动

```bash
cd serving
docker build -t decision-gateway .

# 模型目录挂进来即可（只读）
docker run --gpus all -p 8000:8000 \
  -v /home/sakana/models/decision-models:/models:ro \
  -e MAX_LOADED=1 -e PRELOAD=elephant-1 \
  decision-gateway
```

或者：

```bash
MODEL_ROOT=/home/sakana/models/decision-models docker compose up -d --build
```

## 两种部署形态

- **每模型一容器（推荐）**：`-e MODEL=elephant-1`，容器只服务这一个模型，New API 里加一个 OpenAI 渠道即可。见 [IMAGES.md](IMAGES.md)。
- **一容器多模型**：不设 `MODEL`，靠请求里的 `model` 字段切换，`MAX_LOADED=1` 会换入换出（冷启动要读盘，不适合线上）。

## 调用

除了 `/v1/systemone`，还提供 **OpenAI 兼容的 `/v1/chat/completions`**（支持 `stream`），方便接 new-api / one-api / LiteLLM：

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "elephant-1",
  "messages": [{"role": "user", "content": "{\"state\": {\"ticket\": \"charged twice\"}, \"questions\": {\"tier\": {\"type\": \"choice\", \"criteria\": {\"small\": \"x\", \"mid\": \"y\"}}}}"}]
}'
```

不带 `questions` 时用内置路由预设（`tier` + `escalate`），所以网关的"测试连接"能直接通过。



```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "elephant-1",
  "state": {"ticket": "I was charged twice for order #4821, refund it today."},
  "questions": {
    "tier": {"type": "choice", "instructions": "Which model tier should answer?",
             "criteria": {"small": "lookup/format", "mid": "multi-step reasoning",
                          "frontier": "hard math, long context, ambiguous"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"}
  }
}'
```

```json
{
  "id": "decision-...",
  "object": "structured_decision",
  "model": "elephant-1",
  "answers": {
    "tier": {"type": "choice", "choice": "small", "confidence": 0.87,
             "probabilities": {"small": 0.87, "mid": 0.11, "frontier": 0.02}},
    "urgent": {"type": "noul", "noul": 0.62}
  },
  "usage": {"latency_ms": 84.2}
}
```

`model` 省略时用目录里按字母序的第一个；`GET /v1/models` 列出所有可用模型；`MAX_LOADED=1` 表示同一时刻只驻留一个模型，
切换时自动卸载上一个（31B+26B+9B+4.5B 全驻留需要 ~140GB 显存，不现实）。

## 注意

- **必须 NVIDIA GPU**：三个 runtime 都走 PyTorch CUDA；`DEVICE=cuda:0` 可改。
- **qev 版本敏感**：`serve.patch` 是对 `qev/engine.py` 的上下文补丁，必须 checkout `81ce62f` 后再打，否则 `git apply` 失败。
- **transformers 5.17 是硬要求**：`Gemma4ForConditionalGeneration`、`Qwen3_5*` 都要它。
- **GLiDE 没有视觉**：带 `images` 的请求会 400；Bison/Elephant/Decider 支持图像（Decider 走 `images` 字段）。
- **Bison 的 setup 未随仓库提供**：它和 elephant 的 `readout.json` 完全一致（slot_lm / 256 / 256），共用 qev 路径即可；
  如果 31B 上 qev 报架构不支持，说明 OneJev 这个 commit 的 Gemma4 分支没覆盖到，需要在网关里单独接 `transformers` 的
  `Gemma4ForConditionalGeneration` 做一次前向（readout 头是独立的 `readout.safetensors`，可手动加载）。
- 镜像本身**没有在本机构建**：这台机器连不上 Docker Hub（`registry-1.docker.io` 超时）也没有可用 GPU（NVML 报错），
  所以只做了静态校验 + 网关逻辑单测（假 adapter 跑通 `/v1/systemone`、404、自动选模型、runtime 识别）。

## 离线构建镜像（dockerd 走坏代理时）

这台机器的 dockerd 被固定在 `http://127.0.0.1:7890`（已失效），且 NO_PROXY 不含
`auth.docker.io` 和 blob 跳转域名，所以 `docker pull` 一律 EOF。绕过办法：用
`scripts/pull_image.py` 直连 registry 拉镜像，再 `docker load`。

```bash
# 1. 拉基础镜像（直连，走 1ms.run 镜像源，实测 ~3 MB/s）
python3 scripts/pull_image.py python:3.12-slim -o /tmp/python312.tar
docker load -i /tmp/python312.tar

# 2. 离线构建（wheelhouse + 已备好的 nvidia 库，不联网）
#    Dockerfile.offline 见本目录，构建上下文需要：
#      wheelhouse/  nvidia/  app/  serve.patch  entrypoint.sh
docker build -f Dockerfile.offline -t decision-gateway:local .

# 3. 传到推理机
docker save decision-gateway:local -o /tmp/dg.tar
rsync -a --no-compress /tmp/dg.tar dgx2-fast:/tmp/ && ssh dgx2-fast 'docker load -i /tmp/dg.tar'
```

镜像 6.5 GB，内含 torch 2.14.0+cu130 / transformers 5.17.0 / cuDNN 9.24 / qev。
在 spark-d705 上实测：`--gpus all` 直通正常，GLiDE-0.4B 容器内推理可用。
