# NVFP4 权重量化格式说明（pplx-decider-flash-v1.3）

本文件描述 `pplx-decider-flash-v1.3-NVFP4` checkpoint 里 **NVFP4 权重的物理格式**，
供下游 W4A4 推理 Agent 实现 `NVFP4Linear` / `mm_fp4` 时对齐。

> 本 checkpoint **只包含权重侧量化**。Activation 的在线量化、FP4 GEMM、Linear 替换
> 全部由推理侧实现，不属于本次交付。

---

## 1. 哪些权重被量化

只量化文本 Backbone 内部的 Linear 权重（`language_model.layers.*`），共 3 组：

| 组 | 模块 | 形状 | 层数 | tensor 数 |
|---|---|---|---|---|
| A | `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj` | 9216×2560 / 9216×2560 / 2560×9216 | 32 | 96 |
| B | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | 8192×2560 / 1024×2560 / 1024×2560 / 2560×4096 | 8（full attention） | 32 |
| C | `linear_attn.in_proj_qkv` / `in_proj_z` / `out_proj` | 8192×2560 / 4096×2560 / 2560×4096 | 24（DeltaNet） | 72 |

**保持原精度（BF16）**：`in_proj_a` / `in_proj_b`（小型 gate/decay）、`conv1d`、`A_log`、
`dt_bias`、各种 RMSNorm、`embed_tokens`、`visual.*`（Vision Encoder）、`readout.safetensors`
（Decision Readout）、`locate-span/*`。

逐 tensor 的完整清单见 `nvfp4_manifest.json` 的 `tensors` 字段；允许/排除规则的判定见
`quantization_plan.json`。

---

## 2. Tensor 命名

每个被量化的 Linear 由 3 个 tensor 替代原来的 1 个：

| 后缀 | dtype | 形状 | 含义 |
|---|---|---|---|
| `<prefix>.weight_fp4` | `uint8` | `[N, K/2]` | 打包后的 E2M1 权重（每字节 2 个值，低半字节在前） |
| `<prefix>.weight_scale` | `uint8`（e4m3 的位模式） | `[ceil(N/128)*128 * ceil((K/16)/4)*4]` 一维 | 每 16 个 K 元素一个 block scale，**128x4 swizzled** |
| `<prefix>.weight_global_scale` | `float32` | `[1]` | per-tensor global scale |

其中 `N` = 输出维（`weight.shape[0]`），`K` = 输入维（`weight.shape[1]`，连续维）。

例：
```
language_model.layers.0.mlp.gate_proj.weight      → 原 BF16 [9216, 2560]
language_model.layers.0.mlp.gate_proj.weight_fp4        uint8 [9216, 1280]
language_model.layers.0.mlp.gate_proj.weight_scale      uint8 [1474560]   (= 9216 × 160)
language_model.layers.0.mlp.gate_proj.weight_global_scale  float32 [1]
```

---

## 3. 数值格式

* **权重**：FP4 E2M1（1 符号位 + 2 指数位 + 1 尾数位），可表示
  `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`
* **block scale**：FP8 E4M3（max 448），每 **16** 个连续 K 元素共享一个
* **global scale**：FP32，整个 tensor 共享一个（per-tensor Max PTQ）

码字表（bit3 = sign，低 3 位为幅值）：

| 码字 | 0b000 | 0b001 | 0b010 | 0b011 | 0b100 | 0b101 | 0b110 | 0b111 |
|---|---|---|---|---|---|---|---|---|
| 幅值 | 0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |

### 量化（FlashInfer `nvfp4_quantize`，Max PTQ）

```
amax        = max(|W|)                       # per-tensor, fp32
global_sf   = 448 * 6 / amax                 # = 2688 / amax
block_sf[b] = e4m3( global_sf * amax_block[b] / 6 )
q[i]        = fp4( W[i] * global_sf / block_sf[b] )   # round-half-to-even
```

### 反量化

```
Ŵ[i] = q[i] * block_sf[b] / global_sf
```

---

## 4. 位打包顺序（重要）

一个 `uint8` 装 2 个相邻的 K 元素：

```
byte = (elem[2i+1] << 4) | elem[2i]      # 低半字节 = 偶数下标（前一个），高半字节 = 奇数下标
```

这与 FlashInfer `cast_from_fp4` 的实现一致（`v_2nd = x & 0xF` 放在前、`v_1st = x >> 4` 放在后），
已用真实 kernel 的输出现场核对（`0x07` = 元素 [6.0, 0.0]）。
搞反会得到"每两个元素互换"的错误结果，数值上仍然像模像样，务必用本仓库
`scripts/verify_nvfp4_ckpt.py` 的 `rel_l2` 校验。

---

## 5. block scale 的 128x4 swizzled 布局

`weight_scale` **不是**朴素的 `[N, K/16]` 行优先矩阵，而是 FlashInfer/cutlass NVFP4
GEMM 默认消费的 **128x4 swizzled** 布局（`SfLayout.layout_128x4`，`do_shuffle=False`）。

记：

```
NBLK   = K / 16                       # 每行 block 数
PR     = ceil(N / 128) * 128          # 行方向 padding 到 128 的倍数
PC     = ceil(NBLK / 4) * 4           # 列方向 padding 到 4 的倍数
numel  = PR * PC                      # 这就是 weight_scale 的元素个数
```

逻辑矩阵 `[N, NBLK]` 先 pad 到 `[PR, PC]`（补 0），再按 128 行 × 4 列分块做转置：

```python
# 逻辑布局 [N, NBLK]（e4m3 的 uint8 位模式）
sf = sf_logical                                  # [N, NBLK]
sf = pad_to(sf, PR, PC)
sw = sf.reshape(PR // 128, 4, 32, PC // 4, 4).transpose(1, 3).reshape(PR, PC)
weight_scale = sw.reshape(-1)                    # 写盘形态
```

反变换（读取时用）：

```python
def unswizzle_sf(sf_flat, N, K):
    NBLK = K // 16
    PR, PC = ceil(N / 128) * 128, ceil(NBLK / 4) * 4
    sw = sf_flat.reshape(PR, PC)
    sf = sw.reshape(PR // 128, PC // 4, 32, 4, 4).transpose(1, 3).reshape(PR, PC)
    return sf[:N, :NBLK]
```

本模型的 `N`、`K` 都恰好整除 128 / 64，所以 `PR = N`、`PC = NBLK`，**没有实际 padding**；
上面的公式仍然按通用形式写，下游实现不要假设"一定没有 padding"。

---

## 6. 参考加载代码

```python
import json, torch
from safetensors import safe_open

def load_nvfp4_weight(ckpt_dir, prefix, weight_map, device="cuda"):
    shard = weight_map[f"{prefix}.weight_fp4"]
    with safe_open(f"{ckpt_dir}/{shard}", framework="pt", device="cpu") as f:
        fp4 = f.get_tensor(f"{prefix}.weight_fp4").to(device)          # uint8 [N, K/2]
        sf  = f.get_tensor(f"{prefix}.weight_scale").to(device)        # uint8 [PR*PC]
        gs  = f.get_tensor(f"{prefix}.weight_global_scale").to(device) # float32 [1]
    return fp4, sf, gs          # 直接喂给 FlashInfer mm_fp4 的 b / b_scale / b_global_sf
```

反量化成 BF16 做数值核对（仅调试用，不要用于推理）：

```python
import nvfp4_ref as ref
w_hat = ref.dequantize_swizzled(fp4, sf, gs)   # -> float32 [N, K]
```

---

## 7. 与推理侧对接的约定

1. `mm_fp4` 的 B 矩阵直接使用 `weight_fp4`，`b_scale` 用 `weight_scale`（swizzled
   128x4，`do_shuffle=False`），`b_global_sf` 用 `weight_global_scale`；
   **不要再做一次 swizzle / shuffle**。
2. Activation 侧需要自己的 global scale（`448*6/activation_amax`）与每 16 元素的
   block scale，格式与本文件第 3 节相同；这部分不在本 checkpoint 内。
3. 本 checkpoint 的 `config.json` 里带 `quantization_config`，其中 `weight_only: true`
   明确标注"未包含 activation 量化"。
4. 未量化的 tensor 与原始 BF16 checkpoint **逐位一致**，可以按原逻辑直接加载。
