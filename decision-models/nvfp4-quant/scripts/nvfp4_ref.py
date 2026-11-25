#!/usr/bin/env python3
"""NVFP4 参考实现（与 FlashInfer nvfp4_quantize 语义对齐，只用 PyTorch 基础算子）。

这个模块**不依赖 FlashInfer**，用途有三：
  1. smoke test 中与 FlashInfer 的真实 kernel 交叉验证；
  2. verify 阶段从磁盘重读 checkpoint 后做独立反量化，证明"文件里存的东西
     能被独立读出并解释"；
  3. 量化清单里记录 block scale 的统计量。

格式（与 flashinfer/fp4_quantization.py v0.6.3 的 nvfp4_quantize 一致）：
  - 权重 [M, K]（K 连续）→ packed uint8 [M, K/2]；
    **低半字节 = 偶数下标元素，高半字节 = 奇数下标元素**
    （即 byte = (elem[2i+1] << 4) | elem[2i]，与 FlashInfer cast_from_fp4 一致）；
  - block scale：每 16 个元素一个 fp8_e4m3 缩放因子，逻辑布局 [M, K/16]；
  - global scale：fp32 标量，per-tensor Max PTQ 取 gs = 448 * 6 / amax；
  - 量化：block_sf = e4m3(gs * block_amax / 6)，q = fp4(x * gs / block_sf)；
  - 反量化：x̂ = q * block_sf / gs；
  - scale factor 物理布局（FlashInfer 默认）：128x4 swizzled，
    即把 [padded_M, padded_K16] 视为 [M/128, K16/4, 32, 4, 4] 后 transpose(1, 3)；
    padded_M = ceil(M/128)*128，padded_K16 = ceil(K16/4)*4。
"""
from __future__ import annotations

import torch

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
BLOCK_SIZE = 16

# E2M1 码字 → 数值（bit3 = sign，低 3 位为幅值）
E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

# 幅值码 0..7 的相邻中点；恰好落在中点上时按"码字 round-half-to-even"处理，
# 与 cutlass/hardware 的 __float2e2m1 一致（0.25→0, 0.75→1.0, 1.25→1.0, ...）。
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (0.75, 1.75, 3.5)  # 这些中点上 round-half-even 取较大码字


# --------------------------------------------------------------------------------------
# E2M1 打包 / 解包
# --------------------------------------------------------------------------------------
def fp4_codes(values: torch.Tensor) -> torch.Tensor:
    """浮点值（任意范围）→ E2M1 码字 uint8（未 clamp 时按外插饱和到 6）。"""
    ax = values.abs()
    code = torch.zeros_like(ax, dtype=torch.uint8)
    for mid in _MIDPOINTS:
        code = code + (ax > mid).to(torch.uint8)
    for tie in _TIE_UP:
        code = code + (ax == tie).to(torch.uint8)
    code = code | ((values < 0).to(torch.uint8) << 3)
    return code


def pack_fp4(values: torch.Tensor) -> torch.Tensor:
    """E2M1 数值 → uint8 packed [..., K/2]（高半字节在前）。"""
    codes = fp4_codes(values).reshape(-1)
    if codes.numel() % 2:
        raise ValueError("K 必须是 2 的倍数")
    even, odd = codes[0::2], codes[1::2]   # even -> 低半字节, odd -> 高半字节
    return ((odd << 4) | even).reshape(values.shape[:-1] + (values.shape[-1] // 2,))


def cast_from_fp4(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """uint8 packed [..., K/2] → 数值 [..., K]（低半字节 = 偶数下标）。"""
    lut = E2M1_LUT.to(packed.device)
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=dtype, device=packed.device)
    out[..., 0::2] = lut[lo.long()]
    out[..., 1::2] = lut[hi.long()]
    return out


# --------------------------------------------------------------------------------------
# 参考量化 / 反量化
# --------------------------------------------------------------------------------------
def compute_global_scale(amax: torch.Tensor | float) -> torch.Tensor:
    """per-tensor Max PTQ：gs = 448 * 6 / amax（amax 为 0 时取 1.0，避免 inf）。"""
    a = torch.as_tensor(amax, dtype=torch.float32)
    return torch.where(a > 0, (FP8_E4M3_MAX * FP4_MAX) / torch.clamp(a, min=1e-30), torch.ones_like(a))


def ref_quantize(x: torch.Tensor, global_scale: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    """参考实现：返回 (packed uint8 [M, K/2], block_scale float8_e4m3fn [M, K/16])。"""
    if x.dim() != 2:
        raise ValueError("x 必须是 [M, K] 2D")
    m, k = x.shape
    if k % BLOCK_SIZE:
        raise ValueError(f"K={k} 不是 {BLOCK_SIZE} 的倍数")
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=x.device).reshape(())
    blocks = x.to(torch.float32).reshape(m, k // BLOCK_SIZE, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)
    block_sf = (gs * block_amax / FP4_MAX).to(torch.float8_e4m3fn)
    sf_f32 = block_sf.to(torch.float32)
    scale = torch.where(sf_f32 > 0, gs / sf_f32.clamp_min(1e-30), torch.zeros_like(sf_f32))
    scaled = (blocks * scale.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX).reshape(m, k)
    return pack_fp4(scaled), block_sf


def dequantize(packed: torch.Tensor, block_sf: torch.Tensor, global_scale: torch.Tensor | float) -> torch.Tensor:
    """反量化：x̂ = q * block_sf / gs。packed [M,K/2] uint8，block_sf [M,K/16] e4m3。"""
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=packed.device).reshape(())
    vals = cast_from_fp4(packed, dtype=torch.float32)
    m, k = vals.shape
    sf = block_sf.to(torch.float32)
    scaled = vals.reshape(m, k // BLOCK_SIZE, BLOCK_SIZE) * (sf / gs).unsqueeze(-1)
    return scaled.reshape(m, k)


# --------------------------------------------------------------------------------------
# 128x4 swizzled scale-factor 布局（纯字节搬运，不解释 dtype）
# --------------------------------------------------------------------------------------
def sf_padded_dims(m: int, k: int, block_size: int = BLOCK_SIZE) -> tuple[int, int]:
    """返回 (padded_rows, padded_cols_in_blocks)；padded_cols 按 4 个 block 对齐。"""
    n_blocks = k // block_size
    return ((m + 127) // 128) * 128, ((n_blocks + 3) // 4) * 4


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def swizzle_sf(sf: torch.Tensor, m: int, k: int, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """逻辑布局 [M, K/16]（或已 padded 的 2D/展平向量）→ 128x4 swizzled 展平 uint8。"""
    n_blocks = k // block_size
    padded_rows, padded_cols = sf_padded_dims(m, k, block_size)
    u8 = _as_u8(sf)
    if u8.dim() == 1:
        u8 = u8.reshape(padded_rows, padded_cols)
    else:
        u8 = u8[:m, :n_blocks]
        if u8.shape != (padded_rows, padded_cols):
            u8 = torch.nn.functional.pad(u8, (0, padded_cols - u8.shape[1], 0, padded_rows - u8.shape[0]))
    u8 = u8.contiguous()
    tiles_m, tiles_k = padded_rows // 128, padded_cols // 4
    view = u8.reshape(tiles_m, 4, 32, tiles_k, 4).transpose(1, 3)
    return view.reshape(padded_rows, padded_cols).contiguous().reshape(-1)


def unswizzle_sf(sf_flat: torch.Tensor, m: int, k: int, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """128x4 swizzled 展平 uint8 → 逻辑布局 uint8 [M, K/16]。"""
    n_blocks = k // block_size
    padded_rows, padded_cols = sf_padded_dims(m, k, block_size)
    u8 = _as_u8(sf_flat).reshape(padded_rows, padded_cols)
    tiles_m, tiles_k = padded_rows // 128, padded_cols // 4
    view = u8.reshape(tiles_m, tiles_k, 32, 4, 4).transpose(1, 3)
    full = view.reshape(padded_rows, padded_cols)
    return full[:m, :n_blocks].contiguous()


def dequantize_swizzled(
    packed: torch.Tensor, sf_flat: torch.Tensor, global_scale: torch.Tensor | float
) -> torch.Tensor:
    """直接从 checkpoint 存储的 swizzled scale 反量化。"""
    m, k2 = packed.shape
    sf_e4m3 = unswizzle_sf(sf_flat, m, k2 * 2).view(torch.float8_e4m3fn)
    return dequantize(packed, sf_e4m3, global_scale)


def quant_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    """权重级误差指标（不含任何 forward / GEMM）。"""
    # 用 float64 累加，避免上亿元素下 fp32 的余弦相似度失真
    a = x.to(torch.float64).reshape(-1)
    b = x_hat.to(torch.float64).reshape(-1)
    diff = a - b
    denom = torch.linalg.vector_norm(a).clamp_min(1e-30)
    cos = torch.dot(a, b) / (denom * torch.linalg.vector_norm(b).clamp_min(1e-30))
    return {
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "max_abs_err": float(diff.abs().max()),
        "mean_abs_err": float(diff.abs().mean()),
        "cosine": float(cos),
        "ref_absmax": float(a.abs().max()),
    }
