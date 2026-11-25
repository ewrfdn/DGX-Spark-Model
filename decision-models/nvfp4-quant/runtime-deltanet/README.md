# Decider Flash 的 NVFP4 Prefill 优化

目标模型是 `pplx-decider-flash-v1.3-NVFP4`，Qwen3.5-4B 架构，32 个 decoder 层中有 24 个 Gated DeltaNet 层和 8 个 Full-Attention 层（第 `3/7/11/15/19/23/27/31` 层）。测试与部署在 `local-dgx`（GB10 / SM121）上使用 W4A4 CUTLASS、FlashInfer 0.6.18、FLA 0.6.0、Transformers 5.17.0 和 NVIDIA PyTorch 2.13 / CUDA 13.3。

## 2026-11-26 P0 + P1 Prefill V2 实测与上线

在两项 DeltaNet 融合 + 普通 RMSNorm 基础上，[fused_ops.py](fused_ops.py) 与 [prefill_v2.py](prefill_v2.py) 落地了完整 P0 + P1 优化，并已更新镜像 `decision-nvfp4:0.6.18`（备份 `decision-nvfp4:0.6.18-orig`，标签 `decision-nvfp4:0.6.18-prefill-v2`）部署至 `local-dgx:8016`（`gw-flash-nvfp4-test`，默认 `DECIDER_PREFILL_V2=full`）：

- **P0-1 Split-state Isolated DeltaNet**：预计算一次布局并消除 24×/请求的 Python GPU→CPU 同步，`_isolated_conv4_silu_split_kernel` 直接输出连续 `(q, k, v)`（同时消掉 28 ms FLA `.contiguous()` 拷贝），FLA 对 context 跑 1 遍 `output_final_state=True` 再对问题块跑 `initial_state`，彻底消除多问题 3× 上下文重算。
- **P0-2 Native FlashAttention / `varlen_attn`**：替换因 4D bool mask 退化的 `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80`，8k Full-Attention 提速 3.4–3.9×。
- **P0-3 FLA 原生 16-head GVA**：去掉 `repeat_interleave(2)`，直接传 16-head Q/K + 32-head V。
- **P0-4 Layer 31 末层裁剪**：L31 仅在 1–3 个 readout marker token 上算 Q/Attention/O/Norm/MLP（L31 由 64–76 ms 降至 1.3–2.5 ms）。
- **P1 带宽融合 + 内联 `amax`**：单趟 MLP `silu(gate)*up + amax`、单 CTA `residual_add + rmsnorm + amax`、单趟 Q/K RMSNorm + 局部 RoPE（直读 strided Q 零拷贝）、`attn * sigmoid(gate) + amax`、`gated_rmsnorm + amax` 与逐层共享激活量化（全部 6 个新 Triton 算子单测与 eager 逐位一致，`changed=0`，彻底消灭全部 200 次独立 `AbsMaxOps` 归约）。

### 整模型同进程 A/B（`isolated`，预热 2 对 / 测 6 对）

| 问题数 / 输入 token | 原版基线 P50 | `p1_only`（逐位一致） | `p0_p1_no_prune` | **`p0_p1_full` P50** | 加速 / 吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|---|---:|
| 1 / 2,086 | 395.02 ms | 205.26 ms (1.93×) | 168.72 ms (2.34×) | **166.22 ms** | **2.38× / 12,549 tok/s** | 4.04 GB |
| 1 / 8,078 | 1,812.84 ms | 1,026.15 ms (1.76×) | 710.53 ms (2.55×) | **687.90 ms** | **2.64× / 11,743 tok/s** | 4.80 GB |
| 3 / 2,150 | 829.95 ms | 737.91 ms (1.13×) | 189.38 ms (4.36×) | **186.80 ms** | **4.44× / 11,510 tok/s** | 4.07 GB |
| 3 / 8,142 | 3,573.24 ms | 3,158.77 ms (1.13×) | 769.55 ms (4.64×) | **744.63 ms** | **4.80× / 10,934 tok/s** | 4.93 GB |

### 200 条请求（531 题）双基准回归

| 方案 | 对比原始 BF16 真值（`isolated`） | 对比原始 BF16 真值（`separate`） | 对比原 NVFP4 基线 |
|---|---|---|---|
| 原版 NVFP4 基线 | Top-1 93.03% / MAE 0.0391 / Max 0.480 | Top-1 93.22% / MAE 0.0364 / Max 0.380 | — |
| `p1_only` | Top-1 93.03% / MAE 0.0391 / Max 0.480 | Top-1 93.22% / MAE 0.0364 / Max 0.380 | **531/531 逐位一致（MAE 0.000000）** |
| `p0_p1_no_prune` | Top-1 93.79% / MAE 0.0373 / Max 0.392 | Top-1 93.60% / MAE 0.0356 / Max 0.437 | Top-1 96.61% / 95.48% |
| **`p0_p1_full`** | **Top-1 94.16% / MAE 0.0371 / Max 0.369** | **Top-1 93.41% / MAE 0.0354 / Max 0.438** | Top-1 96.23% / 94.92% |

[完整消融、200 条双基准回归与 32 层 CUPTI 原始数据](reports/prefill_v2_final_20261126.json)

### 线上 HTTP 实测（`local-dgx:8016`，`gw-flash-nvfp4-test`，每组预热 1 次 / 测 5 次）

| 问题数 / 输入 token | 更新前服务端 P50 | **更新后服务端 P50** | 更新后 HTTP 墙钟 P50 | 服务端吞吐（加速） |
|---|---:|---:|---:|---|
| 1 / 1,872 | 346.46 ms | **147.64 ms** | 149.96 ms | **12,680 tok/s (2.35×)** |
| 3 / 1,932 | 736.39 ms | **166.68 ms** | 168.75 ms | **11,591 tok/s (4.42×)** |
| 1 / 7,272 | 1,568.08 ms | **600.14 ms** | 602.82 ms | **12,117 tok/s (2.61×)** |
| 3 / 7,332 | 3,150.04 ms | **654.28 ms** | 657.31 ms | **11,206 tok/s (4.81×)** |

`/health`、`/v1/models`、`/v1/systemone`、`/v1/chat/completions`（非流式与 SSE）及非法题型报错路径（HTTP 500 `RequestError`）均实测通过。[HTTP 实测原始样本](reports/live_http_prefill_v2_20261126.json)

> 开关说明：`DECIDER_PREFILL_V2=full`（默认）/ `no_prune` / `p1`（与原版逐位一致）/ `off`（完全回退原路径）。

---

## 早期阶段：两项 DeltaNet 融合与普通 RMSNorm 记录

早期先融合 DeltaNet 的 gated RMSNorm 和 kernel size 4 的 causal depthwise conv + SiLU。Gated RMSNorm 对齐 ATen 的 FP32 reduction 顺序并保留两次中间 BF16 舍入。卷积按 token-major 布局读写，在 SiLU 前把卷积结果舍入为 BF16。两者都使用镜像现有的 Triton。FLA 递推、W4A4 权重和 GEMM 路径继续沿用原运行时。

## 2026-11-26 性能复测

`local-dgx:8016` 的 `gw-flash-nvfp4-test` 使用原版 CUTLASS W4A4，`isolated`、`fast=False`。本机 HTTP 每组预热 1 次、测 5 次。服务端单问题吞吐为 5,403 / 4,638 tok/s（1,872 / 7,272 token），三问题为 2,624 / 2,328 tok/s（1,932 / 7,332 token）。HTTP P50 比服务端 P50 多约 2–3 ms。[HTTP 原始样本](reports/live_http_20261126.json)

独立容器使用同一个模型实例做交替 A/B。每组预热 2 对、测 6 对。吞吐按请求返回的逻辑输入 token 计算，`isolated` 的重算计入时间。

| 问题数 | 输入 token | 原版 P50 | 两项融合 P50 | 原版 → 融合 tok/s |
|---:|---:|---:|---:|---:|
| 1 | 2,086 | 388.28 ms | 296.03 ms | 5,372 → 7,047 |
| 1 | 8,078 | 1,783.43 ms | 1,424.67 ms | 4,529 → 5,670 |
| 3 | 2,150 | 822.11 ms | 545.71 ms | 2,615 → 3,940 |
| 3 | 8,142 | 3,536.17 ms | 2,445.31 ms | 2,302 → 3,330 |

优化版的 CUDA 活动按 CPU launch 关联到最近的阶段，每个 kernel 或传输事件只统计一次。CUDA 活动用时与表中的墙钟时间分别测量。

| CUDA 阶段 | 单问题 2,086 token | 单问题 8,078 token | 三问题 2,150 token | 三问题 8,142 token |
|---|---:|---:|---:|---:|
| 普通 RMSNorm | 63.8 ms / 22.5% | 281.8 ms / 20.0% | 66.0 ms / 13.3% | 284.0 ms / 12.1% |
| Full attention SDPA | 23.7 ms / 8.3% | 347.5 ms / 24.7% | 29.2 ms / 5.9% | 459.2 ms / 19.6% |
| FLA DeltaNet core | 50.4 ms / 17.8% | 202.1 ms / 14.4% | 152.3 ms / 30.7% | 609.0 ms / 26.0% |
| FP4 GEMM | 59.5 ms / 21.0% | 201.5 ms / 14.3% | 86.9 ms / 17.5% | 308.2 ms / 13.2% |
| FP4 激活量化 | 26.0 ms / 9.2% | 129.4 ms / 9.2% | 42.0 ms / 8.5% | 202.6 ms / 8.7% |

普通 RMSNorm 的 8k 单问题耗时主要在 `mul`（108 ms）、dtype 转换（85 ms）和 `pow`（57 ms）。8k SDPA 仍调用 `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80`。8k 三问题的 FLA core 内部以 `chunk_fwd_kernel_o`（173 ms）和 recurrent state 更新 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（146 ms）最重，Q/K 的 `repeat_interleave` 另占约 62 ms。对照原版的 8k 三问题，gated norm 由 678 ms 降到 58 ms，causal conv 由 446 ms 降到 78 ms。[CUDA 归因与 A/B 原始数据](reports/current_prefill_20261126.json)

## 融合版逐层耗时

两项 DeltaNet 融合保持开启，使用 `isolated` 模式。每档预热 2 次、记录 3 次 CUDA trace。编号从 0 开始，`3、7、11、15、19、23、27、31` 是 full-attention 层。单层 P50 包含该层的 MLP、投影和 norm，CUDA 活动按 launch 归属只计一次。

| 问题数 / 输入 token | 最慢层 | 每个 DeltaNet 层范围 | 每个 full-attention 层范围 | 24 个 DeltaNet 层 P50 合计 | 8 个 full-attention 层 P50 合计 |
|---|---|---:|---:|---:|---:|
| 1 / 2,086 | 3、7 | 7.58–8.52 ms | 10.45–11.44 ms | 187.7 ms | 85.9 ms |
| 1 / 8,078 | 19、23、3 | 32.17–33.51 ms | 74.91–75.90 ms | 791.9 ms | 603.7 ms |
| 3 / 2,150 | 24、4、1 | 16.04–17.05 ms | 11.37–11.80 ms | 394.2 ms | 91.8 ms |
| 3 / 8,142 | 23、15、3 | 65.23–67.89 ms | 87.65–89.34 ms | 1,602.2 ms | 707.9 ms |

8k 单问题里，第 19 层的 75.90 ms 包含约 42.05 ms SDPA 和 13.84 ms 普通 RMSNorm。8k 三问题里，第 22 层是最慢的 DeltaNet 层，67.89 ms 中约 24.82 ms 在 FLA core、11.35 ms 在 FP4 GEMM、7.03 ms 在激活量化。三问题的重算让 24 个 DeltaNet 层合计占约 69% 的 decoder CUDA 时间。[32 层三次样本与分项](reports/decoder_layers_20261126.json)

## 在两项融合之上的普通 RMSNorm A/B

实验版使用 Triton 按 ATen 的 FP32 累加顺序算普通 RMSNorm，再融合缩放和 BF16 输出。6 组算子输入逐位一致。在持续启用 gated norm 与 causal conv 融合的同一进程内，每组预热 1 对、测 4 对，结果如下。

| 问题数 / 输入 token | 原普通 RMSNorm P50 | Triton 普通 RMSNorm P50 | 吞吐提升 |
|---|---:|---:|---:|
| 1 / 2,086 | 298.37 ms | 241.33 ms | 23.6% |
| 1 / 8,078 | 1,439.56 ms | 1,191.51 ms | 20.8% |
| 3 / 2,150 | 547.48 ms | 491.39 ms | 11.4% |
| 3 / 8,142 | 2,472.09 ms | 2,218.27 ms | 11.4% |

两种工作负载各对比 20 条合成回归请求及对应的 2k/8k 计时请求，合计 104 个问题的 logits 都逐位一致。实现和基准脚本可在独立实验进程调用。[整模型 A/B](reports/ab_plain_rmsnorm.json) · [算子微基准](reports/probe_plain_rmsnorm.json)

普通 RMSNorm 融合后按同样的 32 层粒度复测了 8k 请求，各取 3 次 CUDA trace 的 P50。单问题的最慢三层是 3、11、23，约 64.5–64.9 ms/层，其中 SDPA 约 43 ms/层。三问题最慢的是 7、19、27，约 76.5–77.5 ms/层。三问题的 24 个 DeltaNet 层仍合计约 1,488 ms，占 decoder CUDA 时间约 71%，第 22 层的 FLA core 占 25.84 ms。[新增普通 RMSNorm 后的逐层报告](reports/decoder_layers_all_rms_20261126.json)

共享 FP4 激活量化另行在 gated norm + conv 融合的基线上测试。把缓存限制在单个 decoder 层后，吞吐提升 2.1%–2.8%，80 个对比问题 logits 逐位一致。8k 三问题的 PyTorch 峰值分配增加 117 MiB，原来的整次 forward 缓存增加约 16.84 GiB。普通 RMSNorm 与逐层缓存的收益来自独立 A/B。[整次 forward 缓存](reports/shared_quant_ab_20261126.json) · [逐层缓存](reports/shared_quant_layer_ab_20261126.json)

## 整模型 A/B

同一 W4A4 进程中交替跑原版和同时启用两项优化的版本。下表是单问题文本请求的 `decide()` 端到端 P50，包含编码、模型前向和决策读出。每个长度预热 3 对、测量 10 对。

| 模式 | 输入 token | 原版 P50 | 优化 P50 | 原版 → 优化 tok/s |
|---|---:|---:|---:|---:|
| separate | 2,086 | 389.46 ms | 298.41 ms | 5,356 → 6,990 |
| separate | 8,078 | 1,791.91 ms | 1,426.07 ms | 4,508 → 5,665 |
| isolated | 2,086 | 395.91 ms | 304.29 ms | 5,269 → 6,855 |
| isolated | 8,078 | 1,800.49 ms | 1,437.33 ms | 4,487 → 5,620 |

两种模式各使用现有的 200 条合成请求、531 个 `choice` / `noul` / `score` 问题做回归。A/A 和 A/B 都有 531/531 原始 logits 逐位一致、100% Top‑1 一致、概率 MAE 为 0，响应结构和 token usage 一致。这里的参考是同一份 W4A4 checkpoint。逐题汇总和计时样本见 [A/B 原始报告](reports/ab_both.json)。

三问题 `isolated` 请求会额外跑一批重算。按接口报告的输入 token（重算已计入时间）衡量，2,150 token 的 P50 从 818.32 降到 541.13 ms，即 2,627 → 3,973 tok/s。8,142 token 从 3,548.85 降到 2,438.72 ms，即 2,294 → 3,339 tok/s。每档预热 2 对、测量 6 对。见 [三问题 A/B](reports/ab_isolated3.json)。

## DeltaNet 耗时拆解

原来约 76.6% 的统计用 decoder block 做分类，里面也包含 MLP、投影和 norm。量化和 GEMM 按算子分类，计入所属层的时间。本次用 CUDA events 单独给 `linear_attn` 和内部算子计时，下表是每个请求中所有调用合计时间的 P50。`linear_attn` 包含前三项。

| 阶段 | Separate 1 问题，2,086 token | Isolated 3 问题，2,150 token |
|---|---:|---:|
| Gated RMSNorm | 55.43 → 4.70 ms | 170.96 → 14.57 ms |
| Causal conv + SiLU | 37.02 → 6.62 ms | 110.12 → 19.54 ms |
| FLA 递推 | 51.87 → 48.97 ms | 161.80 → 152.68 ms |
| `linear_attn` 总计 | 186.09 → 95.58 ms | 603.31 → 324.89 ms |

优化后的单问题请求里，FLA 递推约 49 ms，仍是 `linear_attn` 内最大的单项。投影和布局处理等合计约 35 ms，DeltaNet decoder 层在 `linear_attn` 外的 input/post RMSNorm 还有约 37 ms。三问题 `isolated` 的两个融合算子各命中 48 次，其中 24 次是 batch=2、长度 2,093 的重算。见 [算子计时和调用形状](reports/profile_deltanet.json)。

## 微基准

微基准使用 checkpoint 中的权重和合成 BF16 激活。Gated RMSNorm 取三个层的权重，覆盖短行数与 batch 1/2 的模型形状。卷积另外用 2,073 token 形状逐层验证全部 24 个权重。两项对比均无 BF16 位差异。

| 算子 | Batch / tokens | 原路径 | Triton |
|---|---:|---:|---:|
| Gated RMSNorm | 1 / 2,073 | 2.26–2.34 ms | 0.219 ms |
| Gated RMSNorm | 1 / 8,121 | 8.91–8.95 ms | 0.813–0.820 ms |
| Causal conv + SiLU | 1 / 2,073 | 1.420 ms | 0.290 ms |
| Causal conv + SiLU | 1 / 8,121 | 5.514 ms | 1.091 ms |
| Causal conv + SiLU | 2 / 2,073 | 2.813 ms | 0.558 ms |

## 运行时接入

在 `nvfp4-inference` 的实验副本中应用 `runtime.patch`，将两个 Triton 模块复制进 `nvfp4/`。补丁基于 `bddf762` 的源码，`DECIDER_DELTANET_OPT` 默认 `off`，可以单独选 `norm`、`conv` 或组合 `both`。它适用 eager 推理和 baseline FLA 递推。遇到不匹配 kernel 约束的张量，运行时走原函数。

`local-dgx` 上已经接好补丁的实验副本位于 `~/experiments/decider-deltanet.2q1WHr`。通过原 `NVFP4Decider.from_pretrained()` 和 `decide()` 跑通三问题 `isolated` smoke，两项融合各命中 48 次。

```bash
DELTA=/path/to/decision-models/nvfp4-quant/runtime-deltanet
cd /path/to/nvfp4-inference-experiment
git apply --check "$DELTA/runtime.patch"
git apply "$DELTA/runtime.patch"
cp "$DELTA/deltanet_norm.py" "$DELTA/deltanet_conv.py" nvfp4/
docker build -f docker/Dockerfile -t decision-nvfp4:deltanet .
IMAGE=decision-nvfp4:deltanet NAME=gw-flash-nvfp4-deltanet PORT=18016 \
  DECIDER_DELTANET_OPT=both ./docker/run_nvfp4.sh
```

## 复测

在具有上述依赖的 GPU 容器内，把运行时仓库根目录放在 `PYTHONPATH`，将此目录、checkpoint 和测试 JSONL 挂载进去。基准默认在同一模型实例内比较 baseline / baseline repeat / candidate，且在每个长度交替测 10 对请求的 P50。

```bash
export PYTHONPATH=/path/to/nvfp4-inference
CKPT=/models/pplx-decider-flash-v1.3-NVFP4
python /bench/benchmark_deltanet.py --checkpoint "$CKPT" \
  --dataset /path/to/nvfp4-inference/data/eval_requests.jsonl \
  --report /out/ab_both.json --optimization both
python /bench/benchmark_deltanet.py --checkpoint "$CKPT" \
  --report /out/ab_isolated3.json --optimization both --mode isolated \
  --isolated-questions 3 --timing-only --warmup 2 --repeat 6
python /bench/profile_deltanet.py --checkpoint "$CKPT" \
  --report /out/profile_deltanet.json --tokens 2000
python /bench/profile_current_prefill.py --checkpoint "$CKPT" \
  --report /out/current_prefill.json
python /bench/profile_decoder_layers.py --checkpoint "$CKPT" \
  --report /out/decoder_layers.json
python /bench/profile_decoder_layers.py --checkpoint "$CKPT" \
  --tokens 8000 --plain-rmsnorm --warmup 1 --report /out/decoder_layers_all_rms.json
python /bench/benchmark_plain_rmsnorm.py --checkpoint "$CKPT" \
  --dataset /path/to/nvfp4-inference/data/eval_requests.jsonl \
  --variant triton_reduction --report /out/ab_plain_rmsnorm.json
python /bench/benchmark_shared_quant.py --checkpoint "$CKPT" \
  --dataset /path/to/nvfp4-inference/data/eval_requests.jsonl \
  --scope layer --warmup 1 --repeat 4 --report /out/shared_quant_layer_ab.json
python /bench/probe_deltanet_norm.py --checkpoint "$CKPT"
python /bench/probe_deltanet_conv.py \
  --weights "$CKPT/model-00001-of-00001.safetensors" --all-weights
```
