"""NVFP4 (W4A4) inference for pplx-decider-flash-v1.3 on NVIDIA GB10 / SM121.

The package only *consumes* the offline weights produced by the quantisation
agent; it never re-quantises weights and never dequantises them into a BF16
production path.  See ``NVFP4_FORMAT.md`` in the checkpoint for the on-disk
format and ``reports/`` for the measurements taken with this code.
"""

from .backend import (
    AMAX_SCALE,
    BACKENDS,
    backend_report,
    quantize_activation,
    resolve_backend,
)
from .checkpoint import Checkpoint, QuantizedModule
from .linear import NVFP4Linear

__all__ = [
    "AMAX_SCALE",
    "BACKENDS",
    "Checkpoint",
    "NVFP4Linear",
    "QuantizedModule",
    "backend_report",
    "quantize_activation",
    "resolve_backend",
]
