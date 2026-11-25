# pplx-decider-flash-v1.3 → NVFP4 离线权重量化

把 `perplexity-ai/pplx-decider-flash-v1.3`（BF16，4.54 B 参数）里 **200 个文本 Backbone Linear 权重**
用 FlashInfer 的 `nvfp4_quantize` 转成**真实 NVFP4**（E2M1 packed weight + FP8 E4M3 block scale +
FP32 per-tensor global scale），其余 523 个 tensor（embedding / norm / DeltaNet 小参数 / vision /
readout）**原样保留**，导出一个可独立读取的量化 checkpoint，交给下游 W4A4 推理 Agent。

> **边界**：本目录只负责**权重**的离线转换。activation 量化、`NVFP4Linear`、`mm_fp4`、模型加载器、
> 任何 forward / GEMM / 性能测试都**不在**本次范围内，也没有被执行过。

| 项目 | 值 |
|---|---|
| 原始模型 | `/home/sakana/models/decision-models/pplx-decider-flash-v1.3`（只读） |
| 量化产物 | `/home/sakana/models/decision-models/pplx-decider-flash-v1.3-NVFP4` |
| 硬件 | NVIDIA GB10（DGX Spark，`compute capability (12, 1)`，132 GB 统一内存） |
| 软件 | torch `2.13.0a0+9186a08b2c.nv26.07` / CUDA `13.3` / FlashInfer `0.6.3` / nvcc `13.3` |
| 量化耗时 | 290 s（读 8.9 GB BF16、写 3.95 GB NVFP4） |
| 校验 | **PASS**（200/200 tensor 重读校验，保留文件 16/16 sha256 一致，抽样 8/8 重新量化按位一致） |

---

## 1. 结论先行

| 指标 | 结果 |
|---|---|
| 量化 tensor | 200（A 组 MLP 96 + B 组 Full-Attention 32 + C 组 DeltaNet 72） |
| 保留 tensor | 523（全部与原始 BF16 **逐位一致**） |
| 量化参数量 | 3.565 B / 4.539 B = **78.5%** |
| checkpoint 体积 | **3.954 GB**（单分片；BF16 原始权重 7.1 GB） |
| 权重级 rel_l2（反量化 vs BF16） | 均值 **9.489%**，最大 9.522%（A 9.495% / B 9.471% / C 9.490%） |
| cosine | ≈ 0.9954 |
| 独立实现交叉验证 | 用 numpy 手工解包（不经过 torch/FlashInfer）复算，rel_l2 与主校验一致到 1e-4 |

> rel_l2 ≈ 9.5% 是 **NVFP4 E2M1 + 16 元素 block** 本身的量化误差量级（block 内 amax 决定网格步长，
> 小幅值元素被压得比较狠），不是解码错误：同一份数据用完全独立的 numpy 解码路径得到同样的数字，
> 且 8 个抽样 tensor 用 FlashInfer 重新量化后 packed/scale **按位相同**。

---

## 2. 目录

```
decision-models/nvfp4-quant/
├── README.md                    # 本文件
├── docker/
│   ├── Dockerfile               # NGC PyTorch 26.07 + flashinfer 0.6.3（离线 wheel）
│   └── fetch_wheels.sh          # 下载 flashinfer wheel（约 7.6 MB）
├── scripts/
│   ├── run.sh                   # 一键入口：probe/inventory/smoke/plan/quantize/verify/docs
│   ├── env_probe.py             # P0 环境指纹（版本 / 签名 / SfLayout / pip freeze hash）
│   ├── inventory_model.py       # P0 输入清单（文件 sha256 / tensor 形状 / 分组统计）
│   ├── smoke_test_nvfp4.py      # P0 算子 smoke test（全零 / 已知值 / 随机 / 真实形状 / 布局 / 确定性）
│   ├── quantize_nvfp4.py        # 主量化脚本（允许清单 → 量化 → 分片 → 索引 → 清单）
│   ├── verify_nvfp4_ckpt.py     # 权重级校验（结构 / 格式 / 数值 / 重读 / 重新量化比对）
│   ├── nvfp4_ref.py             # 与 kernel 语义对齐的参考实现（打包 / 反量化 / swizzle）
│   └── make_manifest_md.py      # 把 nvfp4_manifest.json 渲染成「量化清单.md」
├── docs/
│   ├── NVFP4_FORMAT.md          # 交接用的格式说明（布局 / 位序 / 公式 / 加载代码）
│   ├── 量化清单.md               # 200 个量化 tensor 的逐条明细 + 523 个保留项的归类
│   └── 交接报告.md               # 做了什么 / 怎么验的 / 没做什么 / 下游怎么接
└── reports/                     # 本次实际运行的原始证据（JSON）
    ├── env_report.json
    ├── input_inventory.json
    ├── smoke_report.json
    ├── quantization_plan.json
    ├── nvfp4_manifest.json
    └── verify_report.json
```

---

## 3. 复现

```bash
cd decision-models/nvfp4-quant/docker
./fetch_wheels.sh                              # 只有 flashinfer 本体，7.6 MB
docker build -t nvfp4-quant:0.6.3 .

cd ../scripts
MODEL_DIR=/path/to/pplx-decider-flash-v1.3 \
OUT_DIR=/path/to/pplx-decider-flash-v1.3-NVFP4 \
./run.sh all                                   # probe → inventory → smoke → plan → quantize → verify → docs
```

单步：

```bash
./run.sh probe        # 环境指纹
./run.sh inventory    # 原始模型指纹（纯 CPU）
./run.sh smoke        # 算子 smoke test（唯一会碰 GPU 的"非正式"步骤，仍然不跑 GEMM）
./run.sh plan         # 只产允许/排除清单，不量化
./run.sh quantize     # 正式量化（约 5 分钟）
./run.sh verify       # 权重级校验（约 4 分钟）
```

容器参数：`--gpus all --ipc=host --ulimit memlock=-1 --user $(id -u):$(id -g)`，
模型目录只读挂载，FlashInfer 的 JIT 产物缓存在 `.fi-cache/`（首次编译约 5 s，之后复用）。

---

## 4. 量化范围（允许清单 / 排除清单）

判定完全由 `config.text_config.layer_types` 驱动（24 层 `linear_attention` + 8 层 `full_attention`，
full attention 落在第 3/7/11/15/19/23/27/31 层），脚本里对"层类型不匹配"直接报错退出。

| 组 | tensor | 形状 | 层数 | 数量 | 理由 |
|---|---|---|---|---|---|
| A | `mlp.gate_proj` / `up_proj` / `down_proj` | 9216×2560 / 9216×2560 / 2560×9216 | 32 | 96 | 文本 Backbone 的主体 GEMM |
| B | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | 8192×2560 / 1024×2560 / 1024×2560 / 2560×4096 | 8 | 32 | full attention 投影 |
| C | `linear_attn.in_proj_qkv` / `in_proj_z` / `out_proj` | 8192×2560 / 4096×2560 / 2560×4096 | 24 | 72 | Gated DeltaNet 的大型输入/输出投影 |

保留（BF16，逐位不变，共 523 个）：

- `linear_attn.in_proj_a` / `in_proj_b`（32 维的小型 decay/beta gate）、`conv1d`、`A_log`、`dt_bias`、`norm`
- 所有 RMSNorm（`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm`）
- `embed_tokens`（`tie_word_embeddings`）、`language_model.norm`
- `visual.*`（24 层 Vision Encoder + merger，297 个 tensor）
- `readout.safetensors`（Decision Readout）、`locate-span/*`

> `in_proj_z` 是 4096×2560 的**大**投影（输出门），不是文档里排除的"小型 Gate"，
> 因此纳入 C 组；如果下游想更保守，可用 `--no-deltanet-z` 重跑，它会退回 BF16。

---

## 5. 产物结构

```
pplx-decider-flash-v1.3-NVFP4/
├── model-00001-of-00001.safetensors   # 3.954 GB：1123 个 key
├── model.safetensors.index.json       # 新索引（量化 tensor 是 3 个派生 key）
├── config.json                        # 原 config + quantization_config
├── nvfp4_manifest.json                # 逐 tensor 量化清单（amax / global scale / scale 统计）
├── quantization_plan.json             # 允许/排除清单（723 条判定 + 理由）
├── NVFP4_FORMAT.md                    # 格式说明（给下游）
├── 量化清单.md                         # 人读版清单
├── verify_report.json                 # 本次校验的原始结果
├── env_report.json / smoke_report.json / input_inventory.json
├── readout.safetensors                # 原样保留
├── locate-span/ source/ tokenizer* chat_template.jinja decision_config.json
└── …                                  # 其余非权重文件全部原样复制
```

量化后的 Linear 变成 3 个 tensor：

| key | dtype | shape |
|---|---|---|
| `<prefix>.weight_fp4` | uint8 | `[N, K/2]` |
| `<prefix>.weight_scale` | uint8（e4m3 位模式） | `[PR*PC]`，128x4 swizzled |
| `<prefix>.weight_global_scale` | float32 | `[1]` |

格式细节（位序、swizzle、公式、加载代码）见 [`docs/NVFP4_FORMAT.md`](docs/NVFP4_FORMAT.md)。

---

## 6. 校验做了什么（全部通过）

| 检查 | 方法 | 结果 |
|---|---|---|
| 结构完整 | 索引 ↔ 分片 ↔ 清单三方 key 集合对比 | 1123 key 一致，723 个原始 tensor 无遗漏 |
| 格式合法 | packed dtype/shape、scale 尺寸 = `PR*PC`、e4m3 有限且 ≤448、global scale = 2688/amax | 全通过 |
| 数值 | 从磁盘重读 → 独立参考实现反量化 → 与 BF16 原权重比 rel_l2 / cosine | 200/200，均值 9.489% |
| 产物确实是 FlashInfer 输出 | 抽样 8 个 tensor 用 `nvfp4_quantize` 重新量化，packed 与 scale 按位比较 | 8/8 完全一致 |
| 保留项未被改动 | 523 个 tensor `torch.equal` 逐元素比较 + 16 个非权重文件 sha256 | 全部一致 |
| 可独立读取 | 用 numpy 手工解包（不经 torch/FlashInfer）复算 4 个 tensor | rel_l2 与主校验一致 |

---

## 7. 踩过的坑（下游实现时请务必注意）

1. **半字节顺序**：packed byte 里 **低半字节是偶数下标元素**（`byte = (elem[2i+1] << 4) | elem[2i]`）。
   我最初的实现按"高半字节在前"写，rel_l2 从 0.095 变成 0.40、cos 掉到 0.92——数值上"看起来还行"，
   但结果全错。用 `known_grid` 用例 + `rel_l2` 才能一眼看出来。
2. **`nvfp4_quantize` 返回的 scale 是 2D** `[PR, PC]`（不是文档示例里的一维缓冲），
   落盘时压平成 `[PR*PC]`；读取时按 `[PR, PC]` reshape 后再解 swizzle。
3. **swizzle 方向**：写盘用 `reshape(PR/128, 4, 32, PC/4, 4).transpose(1,3)`，
   读取要用**逆变换** `reshape(PR/128, PC/4, 32, 4, 4).transpose(1,3)`。两者写反不会报错，
   只会让误差从 9.5% 涨到 47%（我第一版 numpy 交叉验证就踩了这个）。
4. **`global_scale` 公式**：`448 * 6 / amax`（per-tensor），amax 为 0 时退化到 1.0。
5. **不要用 `torch._scaled_mm` 或任何 GEMM 去"顺手验证"**——本任务明确禁止，
   权重级 rel_l2 + 重新量化按位比对已经足够证明产物正确。

---

## 8. 已知不在本次范围内

- ❌ activation 在线量化、per-tensor/per-block activation scale 校准
- ❌ `NVFP4Linear`、`mm_fp4`、模型运行时加载器、Module 替换
- ❌ 任何模型 forward / 决策一致性 / API 回归 / 性能（TPS、TTFT、GEMM benchmark）
- ❌ QAT、4over6、SmoothQuant、Gate/Up 合并、算子融合、结构改造
- ✅ 交付的是**面向 W4A4 的 NVFP4 权重**，推理侧兼容性与业务精度由下游 Agent 负责验证

下游 Agent 的接入方式、需要的 activation scale、以及一个最小可用的加载示例，见
[`docs/交接报告.md`](docs/交接报告.md) 与 [`docs/NVFP4_FORMAT.md`](docs/NVFP4_FORMAT.md)。
