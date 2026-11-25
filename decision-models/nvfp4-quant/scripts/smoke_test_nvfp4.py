#!/usr/bin/env python3
"""离线量化 smoke test：只调用 FlashInfer 的 nvfp4_quantize，不跑 GEMM、不跑模型 forward。

覆盖：
  1. 全零矩阵（amax=0 的退化路径）
  2. 已知值矩阵（E2M1 网格上的精确值，验证量化语义与 nibble 顺序）
  3. 随机 BF16 矩阵（统计误差）
  4. 真实权重形状（9216x2560 / 8192x2560 / 2560x9216）
  5. 非 128 整除的行数（120 行，验证 128x4 swizzle 的 padding 行为）
  6. 确定性（同一输入两次量化结果按位一致）

每一项都会与 scripts/nvfp4_ref.py 的独立参考实现交叉校验，
并检查 packed data 的 dtype/shape、block scale 的 e4m3 有效性与物理布局。

用法：
    python smoke_test_nvfp4.py --out smoke_report.json [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import nvfp4_ref as ref  # noqa: E402


def get_api():
    from flashinfer.fp4_quantization import SfLayout, nvfp4_quantize  # 本地安装版本为准

    return nvfp4_quantize, SfLayout


def run_case(name: str, x: torch.Tensor, nvfp4_quantize, SfLayout, expect_exact: bool = False) -> dict:
    device = x.device
    m, k = x.shape
    amax = x.to(torch.float32).abs().amax()
    gs = ref.compute_global_scale(amax).to(device)

    t0 = time.time()
    packed, sf = nvfp4_quantize(
        x, gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False, sf_vec_size=16
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    dt = time.time() - t0

    packed2, sf2 = nvfp4_quantize(
        x, gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False, sf_vec_size=16
    )

    padded_rows, padded_cols = ref.sf_padded_dims(m, k)
    sf_e4m3 = sf.view(torch.float8_e4m3fn).to(torch.float32)
    sf_logical = ref.unswizzle_sf(sf, m, k)
    x_hat = ref.dequantize_swizzled(packed, sf, gs)

    # 参考实现（独立路径）
    ref_packed, ref_sf = ref.ref_quantize(x, gs)
    x_hat_ref = ref.dequantize(ref_packed, ref_sf, gs)

    err = ref.quant_error(x, x_hat)
    err_ref = ref.quant_error(x, x_hat_ref)

    checks: dict[str, dict] = {}
    checks["packed_dtype_uint8"] = {"ok": packed.dtype == torch.uint8, "value": str(packed.dtype)}
    checks["packed_shape"] = {"ok": tuple(packed.shape) == (m, k // 2), "value": list(packed.shape)}
    checks["sf_flat_size"] = {
        "ok": sf.numel() == padded_rows * padded_cols,
        "value": int(sf.numel()),
        "expected": padded_rows * padded_cols,
    }
    checks["sf_finite"] = {"ok": bool(torch.isfinite(sf_e4m3).all()), "value": None}
    checks["sf_nonnegative"] = {"ok": bool((sf_e4m3 >= 0).all()), "value": float(sf_e4m3.min())}
    checks["sf_reciprocal"] = {
        "ok": bool(torch.allclose(ref.swizzle_sf(sf_logical, m, k), sf.view(torch.uint8).reshape(-1), atol=0)),
        "value": "swizzle(unswizzle(sf)) == sf",
    }
    checks["deterministic"] = {
        "ok": bool(torch.equal(packed, packed2) and torch.equal(sf, sf2)),
        "value": None,
    }
    # 与参考实现比对"解出来的码字"：允许极少数元素因舍入边界（round-half-even 的边界值）
    # 不同，但差异比例必须极小，否则说明字节打包顺序或 scale 布局解释错了。
    sf_step = ref.unswizzle_sf(sf, m, k).view(torch.float8_e4m3fn).float() / gs.reshape(())
    ref_step = ref_sf.float() / gs.reshape(())
    q_kernel = x_hat.reshape(m, k // ref.BLOCK_SIZE, ref.BLOCK_SIZE) / sf_step.unsqueeze(-1)
    q_ref = x_hat_ref.reshape(m, k // ref.BLOCK_SIZE, ref.BLOCK_SIZE) / ref_step.unsqueeze(-1)
    mismatch = float(((q_kernel - q_ref).abs() > 0.1).float().mean())
    checks["matches_ref_codes"] = {
        "ok": mismatch < 2e-2,  # 布局/字节序搞反会造成 >40% 不一致，2% 足以区分
        "value": {"mismatch_fraction": mismatch, "max_abs_diff": float((x_hat - x_hat_ref).abs().max())},
    }
    checks["ref_quant_error"] = {"ok": err["rel_l2"] < 0.12, "value": err["rel_l2"]}

    if amax > 0:
        checks["global_scale"] = {
            "ok": abs(float(gs) - 448.0 * 6.0 / float(amax)) < 1e-3 * float(gs),
            "value": float(gs),
            "expected": 448.0 * 6.0 / float(amax),
        }
        nonzero_blocks = ref.unswizzle_sf(ref.swizzle_sf(ref_sf.to(torch.float32), m, k), m, k) > 0
        checks["sf_positive_where_block_nonzero"] = {
            "ok": bool((sf_logical[nonzero_blocks] > 0).all()),
            "value": int((sf_logical[nonzero_blocks] > 0).sum().item()),
        }
    else:
        checks["all_zero_path"] = {
            "ok": bool((packed == 0).all() and (sf == 0).all()),
            "value": {"packed_zero": bool((packed == 0).all()), "sf_zero": bool((sf == 0).all())},
        }

    if expect_exact:
        checks["exact_roundtrip"] = {
            "ok": bool(torch.equal(x_hat.to(x.dtype), x)),
            "value": float((x_hat - x.to(torch.float32)).abs().max()),
        }

    return {
        "case": name,
        "shape": [m, k],
        "dtype_in": str(x.dtype),
        "amax": float(amax),
        "global_scale": float(gs),
        "packed_shape": list(packed.shape),
        "sf_numel": int(sf.numel()),
        "sf_padded": [padded_rows, padded_cols],
        "seconds": dt,
        "quant_error": err,
        "ref_quant_error": err_ref,
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks.values()),
    }


def nibble_order_case(nvfp4_quantize, SfLayout, device) -> dict:
    """验证 packed 字节里"高半字节 = 第一个元素"。

    构造 x[0,0]=6, x[0,1]=0 → 该 block 的 scale 恰好让 6 落在 E2M1 的最大码字 0b0111 上，
    期望 packed[0,0] == 0x07：**低半字节 = 第一个元素**（若搞反会得到 0x70）。
    """
    x = torch.zeros((128, 64), dtype=torch.bfloat16, device=device)
    x[0, 0] = 6.0
    x[0, 1] = 0.0
    gs = ref.compute_global_scale(torch.tensor(6.0, dtype=torch.float32).to(device))
    packed, sf = nvfp4_quantize(x, gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False, sf_vec_size=16)
    byte0 = int(packed[0, 0].item())
    return {
        "case": "nibble_order",
        "byte0": hex(byte0),
        "expected": "0x07",
        "ok": byte0 == 0x07,
        "all_ok": byte0 == 0x07,
        "dequant_first8": ref.cast_from_fp4(packed[0:1, :4])[0].tolist()[:8],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke_report.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("BLOCKED: CUDA 不可用", file=sys.stderr)
        return 2

    nvfp4_quantize, SfLayout = get_api()
    device = torch.device(args.device)
    torch.manual_seed(0)

    cases = []
    cases.append(("zeros_128x256", torch.zeros((128, 256), dtype=torch.bfloat16, device=device), False))

    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    known = grid.repeat(4, 8)  # 每行 64 个元素、每个 16 元素 block 的 amax 都是 6
    known = torch.cat([known, -known], dim=0).to(torch.bfloat16).to(device)  # [8, 64]
    cases.append(("known_grid_8x64", known, True))

    cases.append((
        "random_1024x4096",
        (torch.randn((1024, 4096), dtype=torch.float32) * 0.02).to(torch.bfloat16).to(device),
        False,
    ))
    for shape in ((9216, 2560), (8192, 2560), (2560, 9216), (4096, 2560)):
        cases.append((
            f"real_shape_{shape[0]}x{shape[1]}",
            (torch.randn(shape, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(device),
            False,
        ))
    cases.append((
        "random_120x64_padding",
        (torch.randn((120, 64), dtype=torch.float32) * 0.02).to(torch.bfloat16).to(device),
        False,
    ))

    results = []
    for name, x, exact in cases:
        try:
            results.append(run_case(name, x, nvfp4_quantize, SfLayout, expect_exact=exact))
        except Exception:
            results.append({"case": name, "all_ok": False, "error": traceback.format_exc()})
        del x
        torch.cuda.empty_cache()

    results.append(nibble_order_case(nvfp4_quantize, SfLayout, device))

    all_ok = all(r.get("all_ok", False) for r in results)
    report = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "all_ok": all_ok,
        "cases": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    for r in results:
        status = "PASS" if r.get("all_ok") else "FAIL"
        qe = r.get("quant_error", {})
        print(
            f"[{status}] {r['case']:<26} shape={r.get('shape')} "
            f"rel_l2={qe.get('rel_l2', float('nan')):.5f} "
            f"cos={qe.get('cosine', float('nan')):.6f} "
            f"sf_numel={r.get('sf_numel')}"
        )
        for k, v in (r.get("checks") or {}).items():
            if not v["ok"]:
                print(f"        !! {k}: {v}")
    print(f"\nALL_OK={all_ok}  report={args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
