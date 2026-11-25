"""Unit-test and micro-benchmark every P0/P1 Triton kernel on GB10."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from nvfp4.fused_ops import (
    fused_add_rmsnorm_amax,
    fused_attn_sigmoid_gate_amax,
    fused_gated_rms_norm_amax,
    fused_isolated_conv4_silu,
    fused_qk_norm_partial_rope,
    fused_rmsnorm_amax,
    fused_silu_mul_amax,
)


def bench(fn, warmup=3, rep=15):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], out


def check_tensor(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    eq = torch.equal(got, ref)
    diff = (got.float() - ref.float()).abs()
    changed = int((got != ref).sum().item())
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    print(f"  [{name}] bitwise_eq={eq} changed={changed}/{got.numel()} max_err={max_err:.6g}")


def main() -> None:
    torch.manual_seed(1234)
    device = "cuda"
    T = 8078

    print("=== 1. fused_silu_mul_amax [8078, 9216] ===")
    gate = torch.randn(1, T, 9216, device=device, dtype=torch.bfloat16)
    up = torch.randn(1, T, 9216, device=device, dtype=torch.bfloat16)
    t_ref, (ref_y, ref_amax) = bench(
        lambda: (lambda y: (y, torch.linalg.vector_norm(y, ord=float("inf")).float().reshape(1)))(
            F.silu(gate) * up
        )
    )
    t_opt, (opt_y, opt_amax) = bench(lambda: fused_silu_mul_amax(gate, up))
    check_tensor("out", opt_y, ref_y)
    check_tensor("amax", opt_amax, ref_amax)
    print(f"  latency: {t_ref:.3f} -> {t_opt:.3f} ms ({t_ref/t_opt:.2f}x) | 32L: {t_ref*32:.1f} -> {t_opt*32:.1f} ms")

    # Edge cases: all-zero, -inf, NaN
    z_out, z_amax = fused_silu_mul_amax(torch.zeros_like(gate[:1, :16]), torch.zeros_like(up[:1, :16]))
    assert float(z_amax.item()) == 0.0, f"Expected 0 amax, got {z_amax}"
    nan_g = gate[:1, :16].clone()
    nan_g[0, 3, 7] = float("nan")
    _, nan_amax = fused_silu_mul_amax(nan_g, up[:1, :16])
    assert torch.isnan(nan_amax).all(), f"Expected NaN amax, got {nan_amax}"
    print("  [edge cases zero & NaN] PASS")

    print("\n=== 2. fused_rmsnorm_amax & fused_add_rmsnorm_amax [8078, 2560] ===")
    res = torch.randn(1, T, 2560, device=device, dtype=torch.bfloat16)
    delta = torch.randn(1, T, 2560, device=device, dtype=torch.bfloat16)
    w = torch.randn(2560, device=device, dtype=torch.bfloat16) * 0.1
    eps = 1e-6

    def eager_add_rms():
        h = res + delta
        hf = h.float()
        y = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(torch.bfloat16)
        am = torch.linalg.vector_norm(y, ord=float("inf")).float().reshape(1)
        return h, y, am

    t_ref, (ref_h, ref_n, ref_am) = bench(eager_add_rms)
    t_opt, (opt_h, opt_n, opt_am) = bench(lambda: fused_add_rmsnorm_amax(res, delta, w, eps))
    check_tensor("sum_h", opt_h, ref_h)
    check_tensor("norm_y", opt_n, ref_n)
    check_tensor("amax", opt_am, ref_am)
    print(f"  add+rms+amax latency: {t_ref:.3f} -> {t_opt:.3f} ms ({t_ref/t_opt:.2f}x) | 64x: {t_ref*64:.1f} -> {t_opt*64:.1f} ms")

    t_rn, (opt_rn, opt_ram) = bench(lambda: fused_rmsnorm_amax(ref_h, w, eps))
    check_tensor("standalone rms_y", opt_rn, ref_n)
    check_tensor("standalone amax", opt_ram, ref_am)
    print(f"  standalone rms+amax latency: {t_rn:.3f} ms")

    print("\n=== 3. fused_qk_norm_partial_rope (Q strided [1, 8078, 16, 256] & K [1, 8078, 4, 256]) ===")
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    qg = torch.randn(1, T, 16, 2, 256, device=device, dtype=torch.bfloat16)
    q_raw = qg[:, :, :, 0, :]
    gate_raw = qg[:, :, :, 1, :]
    w_q = torch.randn(256, device=device, dtype=torch.bfloat16) * 0.1
    cos = torch.randn(1, T, 64, device=device, dtype=torch.bfloat16)
    sin = torch.randn(1, T, 64, device=device, dtype=torch.bfloat16)

    def eager_q_rope():
        xf = q_raw.float()
        qn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w_q.float())).to(torch.bfloat16)
        qt = qn.transpose(1, 2)
        qr, _ = apply_rotary_pos_emb(qt, qt, cos, sin)
        return qr.transpose(1, 2).contiguous()

    t_ref, ref_qr = bench(eager_q_rope)
    t_opt, opt_qr = bench(lambda: fused_qk_norm_partial_rope(q_raw, w_q, cos, sin, eps))
    check_tensor("q_rope", opt_qr, ref_qr)
    print(f"  Q norm+rope latency: {t_ref:.3f} -> {t_opt:.3f} ms ({t_ref/t_opt:.2f}x) | 8L: {t_ref*8:.1f} -> {t_opt*8:.1f} ms")

    k_raw = torch.randn(1, T, 4, 256, device=device, dtype=torch.bfloat16)
    def eager_k_rope():
        xf = k_raw.float()
        kn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w_q.float())).to(torch.bfloat16)
        kt = kn.transpose(1, 2)
        kr, _ = apply_rotary_pos_emb(kt, kt, cos, sin)
        return kr.transpose(1, 2).contiguous()

    t_ref_k, ref_kr = bench(eager_k_rope)
    t_opt_k, opt_kr = bench(lambda: fused_qk_norm_partial_rope(k_raw, w_q, cos, sin, eps))
    check_tensor("k_rope", opt_kr, ref_kr)
    print(f"  K norm+rope latency: {t_ref_k:.3f} -> {t_opt_k:.3f} ms ({t_ref_k/t_opt_k:.2f}x) | 8L: {t_ref_k*8:.1f} -> {t_opt_k*8:.1f} ms")

    print("\n=== 4. fused_attn_sigmoid_gate_amax [1, 8078, 16, 256] ===")
    attn_out = torch.randn(1, T, 16, 256, device=device, dtype=torch.bfloat16)
    def eager_gate():
        y = attn_out.reshape(1, T, -1).contiguous() * torch.sigmoid(gate_raw.reshape(1, T, -1))
        am = torch.linalg.vector_norm(y, ord=float("inf")).float().reshape(1)
        return y, am

    t_ref, (ref_g, ref_gam) = bench(eager_gate)
    t_opt, (opt_g, opt_gam) = bench(lambda: fused_attn_sigmoid_gate_amax(attn_out, gate_raw))
    check_tensor("gated", opt_g, ref_g)
    check_tensor("amax", opt_gam, ref_gam)
    print(f"  attn_gate+amax latency: {t_ref:.3f} -> {t_opt:.3f} ms ({t_ref/t_opt:.2f}x) | 8L: {t_ref*8:.1f} -> {t_opt*8:.1f} ms")

    print("\n=== 5. fused_gated_rms_norm_amax [1, 8078, 32, 128] ===")
    core = torch.randn(1, T, 32, 128, device=device, dtype=torch.bfloat16)
    z = torch.randn(1, T, 32, 128, device=device, dtype=torch.bfloat16)
    w_dn = torch.randn(128, device=device, dtype=torch.bfloat16)
    def eager_gn():
        h2 = core.reshape(-1, 128).float()
        v = h2.pow(2).mean(-1, keepdim=True)
        n = (h2 * torch.rsqrt(v + eps)).to(torch.bfloat16)
        y = ((w_dn * n) * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(1, T, 4096)
        am = torch.linalg.vector_norm(y, ord=float("inf")).float().reshape(1)
        return y, am

    t_ref, (ref_gn, ref_gnam) = bench(eager_gn)
    t_opt, (opt_gn, opt_gnam) = bench(lambda: fused_gated_rms_norm_amax(core, z, w_dn, eps, out_shape=(1, T, 4096)))
    check_tensor("gated_norm", opt_gn, ref_gn)
    check_tensor("amax", opt_gnam, ref_gnam)
    print(f"  gated_norm+amax latency: {t_ref:.3f} -> {t_opt:.3f} ms ({t_ref/t_opt:.2f}x) | 24L: {t_ref*24:.1f} -> {t_opt*24:.1f} ms")

    print("\n=== 6. fused_isolated_conv4_silu (isolated Q=3 ctx=8046 + 3x32) ===")
    ctx = 8046
    T3 = ctx + 96
    x_qkv = torch.randn(1, T3, 8192, device=device, dtype=torch.bfloat16)
    w_conv = torch.randn(8192, 4, device=device, dtype=torch.bfloat16) * 0.2
    prev = torch.full((4, T3), -1, dtype=torch.int32)
    prev[3, :ctx] = torch.arange(0, ctx, dtype=torch.int32)
    for s in range(1, 4):
        prev[s - 1, s:ctx] = torch.arange(0, ctx - s, dtype=torch.int32)
    for blk in range(3):
        b = ctx + blk * 32
        e = b + 32
        prev[3, b:e] = torch.arange(b, e, dtype=torch.int32)
        for s in range(1, 4):
            prev[s - 1, b + s : e] = torch.arange(b, e - s, dtype=torch.int32)
            for off in range(s):
                prev[s - 1, b + off] = ctx + off - s
    prev_cuda = prev.to(device)

    # Reference: run F.conv1d separately on [ctx+q0], [ctx+q1], [ctx+q2]
    def ref_iso_conv():
        outs = []
        w3 = w_conv.unsqueeze(1)
        c0 = F.silu(F.conv1d(x_qkv[:, : ctx + 32].transpose(1, 2), w3, padding=3, groups=8192)[..., : ctx + 32]).transpose(1, 2)
        outs.append(c0)
        for blk in (1, 2):
            b = ctx + blk * 32
            seq = torch.cat((x_qkv[:, :ctx], x_qkv[:, b : b + 32]), dim=1)
            c = F.silu(F.conv1d(seq.transpose(1, 2), w3, padding=3, groups=8192)[..., : ctx + 32]).transpose(1, 2)
            outs.append(c[:, ctx:])
        return torch.cat(outs, dim=1)

    ref_ic = ref_iso_conv()
    t_ic, opt_ic = bench(lambda: fused_isolated_conv4_silu(x_qkv, w_conv, prev_cuda))
    check_tensor("isolated_conv4_silu", opt_ic, ref_ic)
    print(f"  isolated_conv4_silu latency: {t_ic:.3f} ms | 24L: {t_ic*24:.1f} ms")


if __name__ == "__main__":
    main()
