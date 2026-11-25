"""Compare the Triton prefill convolution with eager CUDA on GB10."""

import argparse
from pathlib import Path
from statistics import median
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from safetensors import safe_open
import torch
from torch.nn import functional as F

from nvfp4.deltanet_conv import causal_conv4_silu


def eager_conv(x, weight):
    return F.silu(F.conv1d(x, weight.unsqueeze(1), padding=3, groups=x.shape[1])[:, :, :x.shape[2]])


def measure_ms(run, repeats):
    for _ in range(3):
        run()
    durations = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        finish.record()
        finish.synchronize()
        durations.append(start.elapsed_time(finish))
    return median(durations)


def validate_checkpoint_layers(path):
    batch, length, channels = 1, 2073, 8192
    x = torch.randn((batch, length, channels), dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    x[:, :64, :] *= 16
    x[:, 64:128, :] *= 0.0625
    with safe_open(path, framework="pt", device=0) as shard:
        names = [name for name in shard.keys() if name.endswith("linear_attn.conv1d.weight")]
        mismatches = {}
        for name in names:
            weight = shard.get_tensor(name).squeeze(1)
            expected = eager_conv(x, weight)
            actual = causal_conv4_silu(x, weight)
            count = torch.count_nonzero(expected.view(torch.int16) != actual.view(torch.int16)).item()
            if count:
                mismatches[name] = count
    print(f"checkpoint_layers={len(names)} values_per_layer={x.numel()} bit_mismatches={mismatches}", flush=True)
    return bool(names) and not mismatches


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="NVFP4 safetensors shard with preserved BF16 convolution weights")
    parser.add_argument("--all-weights", action="store_true", help="Also compare every Gated DeltaNet layer against eager")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.all_weights and args.weights is None:
        parser.error("--all-weights requires --weights")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    torch.manual_seed(17)
    if args.weights:
        with safe_open(args.weights, framework="pt", device=0) as shard:
            name = next(key for key in shard.keys() if key.endswith("linear_attn.conv1d.weight"))
            weight = shard.get_tensor(name).squeeze(1)
        print(name, flush=True)
    else:
        weight = torch.randn((8192, 4), dtype=torch.bfloat16, device="cuda") * 0.25

    shapes = (
        (2, 2, 129),
        (1, 7, 129),
        (2, 33, 8192),
        (1, 2073, 8192),
        (1, 8121, 8192),
        (2, 2073, 8192),
    )
    print("B,T,C,eager_ms,triton_ms,speedup,bit_mismatch,total,max_abs,rel_l2", flush=True)
    passed = True
    for batch, length, channels in shapes:
        shape = (batch, length + 2, channels)
        # Exercise both left padding and an input view with a batch stride larger than T * C.
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")[:, 1:-1, :]
        x[:, length // 2, :] = 0
        x = x.transpose(1, 2)
        current_weight = weight[:channels]
        expected = eager_conv(x, current_weight)
        actual = causal_conv4_silu(x, current_weight)
        bit_mismatch = torch.count_nonzero(expected.view(torch.int16) != actual.view(torch.int16)).item()
        passed = passed and bit_mismatch == 0
        difference = expected.float() - actual.float()
        max_abs = difference.abs().max().item()
        rel_l2 = (difference.norm() / expected.float().norm()).item()
        del actual, expected, difference
        eager_ms = measure_ms(lambda: eager_conv(x, current_weight), args.repeats)
        triton_ms = measure_ms(lambda: causal_conv4_silu(x, current_weight), args.repeats)
        print(
            f"{batch},{length},{channels},{eager_ms:.4f},{triton_ms:.4f},{eager_ms / triton_ms:.3f},"
            f"{bit_mismatch},{x.numel()},{max_abs:.6g},{rel_l2:.6g}",
            flush=True,
        )
    if args.all_weights:
        torch.manual_seed(302)
        passed = validate_checkpoint_layers(args.weights) and passed
    if not passed:
        raise SystemExit("The Triton convolution differed from eager output")


if __name__ == "__main__":
    main()
