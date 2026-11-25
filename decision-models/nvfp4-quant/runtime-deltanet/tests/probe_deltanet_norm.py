"""Check fused gated RMSNorm against the installed Transformers eager implementation."""

import argparse
import json
import statistics
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

from nvfp4.deltanet_norm import install_deltanet_norm


def time_cuda(run, runs=15):
    for _ in range(5):
        run()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(runs):
        start.record()
        run()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(376)
    checkpoint = args.checkpoint
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    weights = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    module = Qwen3_5RMSNormGated(128, eps=config["rms_norm_eps"]).to(device="cuda", dtype=torch.bfloat16).eval()
    baseline = module.forward
    install_deltanet_norm(module)
    cases = []
    for layer in (0, 14, 30):
        key = f"language_model.layers.{layer}.linear_attn.norm.weight"
        with safe_open(checkpoint / weights[key], framework="pt", device="cuda:0") as shard:
            module.weight.copy_(shard.get_tensor(key))
        for rows in (7, 16, 35, 2073 * 32, 2073 * 64, 8121 * 32):
            hidden = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16).mul_(0.0625)
            hidden[0].zero_()
            gate = torch.randn_like(hidden).mul_(2)
            reference = baseline(hidden, gate)
            candidate = module(hidden, gate)
            mismatch = torch.count_nonzero(reference.view(torch.int16) != candidate.view(torch.int16)).item()
            case = {"layer": layer, "rows": rows, "elements": hidden.numel(), "bit_mismatches": mismatch}
            if rows >= 2073 * 32:
                case["eager_ms"] = time_cuda(lambda: baseline(hidden, gate))
                case["fused_ms"] = time_cuda(lambda: module(hidden, gate))
            print(json.dumps(case), flush=True)
            cases.append(case)
    if any(case["bit_mismatches"] for case in cases):
        raise SystemExit("Gated RMSNorm bitwise comparison failed")


if __name__ == "__main__":
    main()
