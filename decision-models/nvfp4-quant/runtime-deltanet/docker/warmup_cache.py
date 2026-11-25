"""Warm the FlashInfer/CuTe-DSL on-disk cache for the NVFP4 decider.

Run once with a persistent ``FLASHINFER_WORKSPACE_BASE`` volume so that
container starts avoid the multi-minute JIT compile of the FP4 quantizer and
the b12x kernels.  This is a build/deploy step, not part of a request.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", default="128,560,1060,2066,4076,8092")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.checkpoint) / "source" / "src"))
    from nvfp4.decider import NVFP4Decider

    decider = NVFP4Decider.from_pretrained(args.checkpoint, device="cuda", fast=False)
    tokenizer = decider.processor.tokenizer
    paragraph = (
        "The customer reports an intermittent 401 after key rotation and duplicated billing records. "
        "They attached gateway logs and asked for the next step. "
    )
    for target in [int(x) for x in args.tokens.split(",")]:
        state = paragraph
        while len(tokenizer.encode(state, add_special_tokens=False)) < target:
            state += paragraph
        request = {
            "state": state,
            "questions": {"escalate": {"type": "noul", "instructions": "Does this need a human?"}},
        }
        decider.decide(request)
        torch.cuda.synchronize()
        print(f"warmed {target}", flush=True)
    print("warmup done", flush=True)


if __name__ == "__main__":
    main()
