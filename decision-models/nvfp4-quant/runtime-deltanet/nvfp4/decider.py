"""Build a ``Decider`` whose text-backbone Linears are NVFP4, without BF16 weights.

The original ``pplx_decider.Decider`` is subclassed so that every request /
response / mask / DeltaNet / readout behaviour is inherited unchanged; only
``__init__`` differs, because it must build the backbone from the *packed*
checkpoint instead of ``Qwen3_5Model.from_pretrained``.

Load sequence
-------------
1. ``AutoConfig`` + ``Qwen3_5Model(config)`` under ``no_init_weights`` (CPU
   skeleton, no random initialisation cost).
2. Replace every manifest target Linear with an ``NVFP4Linear`` (meta buffers).
3. Read all tensors from the NVFP4 safetensors shard(s) straight to the GPU,
   renaming ``weight_fp4``/``weight_scale`` to the module's buffers.
4. ``load_state_dict(assign=True)``; require zero unexplained missing/unexpected
   keys and zero remaining meta tensors.
5. Move residual non-persistent buffers (RoPE inverse frequencies) to the GPU
   with ``module.to(device)`` — no dtype cast, so FP8/FP32 buffers stay intact.
"""

from __future__ import annotations

import json
import math
import os
import string
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor
from transformers.initialization import no_init_weights
from safetensors.torch import load_file

from pplx_decider.model import (
    DEFAULT_MAX_LENGTH,
    Decider,
    IsolatedQuestions,
    FastOptions,
    enable_noncausal_full_attention,
)
from pplx_decider.protocol import (
    LEGACY_MAX_PIXELS,
    MAX_OPTIONS,
    MIN_PIXELS,
    MODES,
    PROTOCOL,
    Encoder,
    load_json,
)

from .backend import backend_report
from .checkpoint import Checkpoint
from .linear import NVFP4Linear


def _patch_locate_lora() -> None:
    """Let Locate's LoRA wrapper see ``NVFP4Linear`` (no BF16 ``weight`` tensor)."""
    try:
        from pplx_decider import locate as _locate
    except Exception:  # pragma: no cover - Locate is optional
        return
    if getattr(_locate, "_nvfp4_patched", False):
        return
    original = _locate.LoRALinear.__init__

    def init(self: Any, base: nn.Module, rank: int, alpha: float) -> None:
        if isinstance(base, NVFP4Linear):
            nn.Module.__init__(self)
            self.base = base
            self.scale = alpha / rank
            device = base.weight_packed.device
            self.lora_a = nn.Parameter(
                torch.zeros(rank, base.in_features, device=device), requires_grad=False
            )
            self.lora_b = nn.Parameter(
                torch.zeros(base.out_features, rank, device=device), requires_grad=False
            )
            self.enabled = False
            return
        original(self, base, rank, alpha)

    _locate.LoRALinear.__init__ = init

    # Locate verifies its base checkpoint by sha256.  The NVFP4 artifact adds
    # ``quantization_config`` to config.json, so the original hash cannot match;
    # verify every other file exactly and the original config hash through the
    # untouched release-manifest.json (and the BF16 config when mounted).
    original_adapter_init = _locate.LocateAdapter.__init__

    def adapter_init(self: Any, decider: Any, adapter: Any, *, verify_base: bool = True) -> None:
        if verify_base and isinstance(decider, NVFP4Decider):
            _verify_locate_base(decider, Path(adapter))
            verify_base = False
        original_adapter_init(self, decider, adapter, verify_base=verify_base)

    _locate.LocateAdapter.__init__ = adapter_init
    _locate._nvfp4_patched = True


def _verify_locate_base(decider: "NVFP4Decider", adapter: Path) -> None:
    """Locate's base-integrity check, adapted to the NVFP4 checkpoint."""
    from .checkpoint import sha256_file

    config = json.loads((adapter / "locate_config.json").read_text())
    expected: dict[str, str] = config.get("base", {}).get("sha256", {})
    release = json.loads((decider.checkpoint / "release-manifest.json").read_text())
    for name, want in expected.items():
        path = decider.checkpoint / name
        if name == "config.json":
            original = release.get("files", {}).get("config.json", {}).get("sha256")
            if original != want:
                raise ValueError(
                    "Locate adapter base config hash does not match the checkpoint's release manifest"
                )
            base = os.environ.get("DECIDER_NVFP4_BASE_CHECKPOINT")
            if base and (Path(base) / "config.json").is_file():
                if sha256_file(Path(base) / "config.json") != want:
                    raise ValueError("Locate adapter was trained on another base: config.json differs")
            continue
        if not path.is_file() or sha256_file(path) != want:
            raise ValueError(f"Locate adapter was trained on another base: {name} differs")


def _replace_linears(model: nn.Module, checkpoint: Checkpoint, device: torch.device) -> int:
    replaced = 0
    for prefix, module in checkpoint.quantized.items():
        parent_path, _, leaf = prefix.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        current = getattr(parent, leaf, None)
        if current is None:
            raise ValueError(f"manifest target {prefix!r} is not a module of the backbone")
        if isinstance(current, nn.Linear) and not isinstance(current, NVFP4Linear):
            if current.in_features != module.k or current.out_features != module.n:
                raise ValueError(
                    f"{prefix}: model Linear is {current.out_features}x{current.in_features}, "
                    f"manifest says {module.n}x{module.k}"
                )
        else:
            raise ValueError(f"{prefix}: expected an nn.Linear in the skeleton, got {type(current).__name__}")
        with torch.device("meta"):
            replacement = NVFP4Linear(
                module.k,
                module.n,
                torch.empty((module.n, module.k // 2), dtype=torch.uint8),
                torch.empty((module.scale_padded[0] * module.scale_padded[1],), dtype=torch.uint8),
                torch.empty((1,), dtype=torch.float32),
                prefix=prefix,
            )
        setattr(parent, leaf, replacement)
        replaced += 1
    return replaced


def build_nvfp4_backbone(
    checkpoint: Checkpoint,
    hf_config: Any,
    device: torch.device,
    *,
    validate: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct the Qwen3_5 backbone, replace targets, load packed weights."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    started = time.perf_counter()
    validation = checkpoint.validate(check_headers=validate) if validate else {}

    with no_init_weights():
        model = Qwen3_5Model(hf_config)
    replaced = _replace_linears(model, checkpoint, device)

    state, load_stats = checkpoint.load_state_dict(device)
    missing, unexpected = model.load_state_dict(state, assign=True, strict=False)
    del state
    if missing or unexpected:
        raise ValueError(
            "NVFP4 backbone load has unexplained diffs: "
            f"missing={sorted(missing)[:10]} ({len(missing)}), "
            f"unexpected={sorted(unexpected)[:10]} ({len(unexpected)})"
        )
    left_meta = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if left_meta:
        raise ValueError(f"uninitialised meta tensors after NVFP4 load: {left_meta[:10]}")

    # Materialise non-persistent buffers (RoPE inv_freq) on the device; dtypes are preserved.
    model.to(device)
    model.requires_grad_(False).eval()

    stats = {
        "validation": validation,
        "modules_replaced": replaced,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "load": load_stats,
        "seconds": round(time.perf_counter() - started, 2),
        "backends": sorted({m.backend for m in model.modules() if isinstance(m, NVFP4Linear)}),
    }
    return model, stats


class NVFP4Decider(Decider):
    """A ``Decider`` running its text-backbone projections as native W4A4 NVFP4."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cuda",
        fast: bool | FastOptions = False,
        mode: str = "separate",
        max_length: int | None = DEFAULT_MAX_LENGTH,
        max_pixels: int | None = None,
        batch_tokens: int = 32768,
        cpu_threads: int = 8,
        locate: bool | None = None,
        validate: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._inference_lock = threading.RLock()
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The NVFP4 path requires a CUDA device")
        self.mode = mode
        protocol = load_json(checkpoint / "joint_config.json") if (checkpoint / "joint_config.json").exists() else {}
        if protocol != PROTOCOL:
            raise ValueError("This checkpoint does not declare the chunked joint protocol (version 4)")
        config = load_json(checkpoint / "decision_config.json")
        if config.get("attention_mode") != "noncausal_full_attention" or config.get("pooling", "last") != "last":
            raise ValueError("Expected a noncausal checkpoint with per-position readouts")
        self.temperature = float(config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Saved temperature must be positive and finite")
        self.codes: list[str] = list(config["codes"])
        self.max_length = max_length
        self.batch_tokens = batch_tokens
        self.max_pixels = int(max_pixels or config.get("max_pixels") or LEGACY_MAX_PIXELS)

        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": self.max_pixels}
        tokenizer = self.processor.tokenizer
        if len(self.codes) != MAX_OPTIONS or any(
            tokenizer.encode(code, add_special_tokens=False) != [token_id]
            for code, token_id in zip(self.codes, config["token_ids"], strict=True)
        ):
            raise ValueError("Checkpoint answer codes differ from its tokenizer")
        if any(not set(code) <= set(string.ascii_uppercase) for code in self.codes):
            raise ValueError("Unexpected answer codes")
        self.encoder = Encoder(self.processor, self.codes)

        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(str(checkpoint))
        nvfp4_checkpoint = Checkpoint(checkpoint)
        self.nvfp4_checkpoint = nvfp4_checkpoint
        self.backbone, self.load_report = build_nvfp4_backbone(
            nvfp4_checkpoint, hf_config, self.device, validate=validate
        )
        self.is_moe = False
        hidden = self.backbone.config.text_config.hidden_size
        self.readout = nn.Linear(hidden, MAX_OPTIONS, bias=False, dtype=torch.bfloat16)
        self.readout.load_state_dict(load_file(str(checkpoint / "readout.safetensors")))
        text_model = self.backbone.language_model
        enable_noncausal_full_attention(text_model)
        self.isolation = IsolatedQuestions(text_model)
        self.readout.requires_grad_(False).eval().to(self.device)

        self.gdn_backend = os.environ.get("DECIDER_GDN_BACKEND", "baseline").lower()
        if self.gdn_backend not in {"baseline", "flashinfer"}:
            raise ValueError(f"Unknown DECIDER_GDN_BACKEND={self.gdn_backend!r}; expected baseline/flashinfer")

        self.prefill_v2 = os.environ.get("DECIDER_PREFILL_V2", "off").strip().lower()
        self.prefill_opt = os.environ.get("DECIDER_PREFILL_OPT", "0") == "1"
        self.deltanet_opt = os.environ.get("DECIDER_DELTANET_OPT", "off").lower()
        if self.deltanet_opt not in {"off", "norm", "conv", "both"}:
            raise ValueError("DECIDER_DELTANET_OPT must be off, norm, conv, or both")

        from .prefill_v2 import install_prefill_v2, prefill_v2_config

        v2_cfg = prefill_v2_config(self.prefill_v2)
        if v2_cfg is not None:
            if fast or self.gdn_backend != "baseline":
                raise ValueError("DECIDER_PREFILL_V2 requires eager inference with the baseline GDN core")
            install_prefill_v2(self, v2_cfg)
        else:
            if self.prefill_opt:
                self.deltanet_opt = "both"
            if self.deltanet_opt != "off" and (fast or self.gdn_backend != "baseline"):
                flag = "DECIDER_PREFILL_OPT" if self.prefill_opt else "DECIDER_DELTANET_OPT"
                raise ValueError(f"{flag} requires eager inference with the baseline GDN core")
            if self.gdn_backend == "flashinfer":
                from .gdn import install_flashinfer_gdn

                text_config = self.backbone.config.text_config
                ratio = int(text_config.linear_num_value_heads) // int(text_config.linear_num_key_heads)
                install_flashinfer_gdn(gva_ratio=max(1, ratio))
            if self.prefill_opt:
                from .prefill_optimizations import install_prefill_optimizations

                install_prefill_optimizations(self.backbone)
            else:
                if self.deltanet_opt in {"norm", "both"}:
                    from .deltanet_norm import install_deltanet_norm

                    install_deltanet_norm(self.backbone)
                if self.deltanet_opt in {"conv", "both"}:
                    from .deltanet_conv import install_deltanet_conv

                    install_deltanet_conv(self.backbone)

        self.fast = None
        if fast:
            from pplx_decider.fast import FastPath

            self.fast = FastPath(self, fast if isinstance(fast, FastOptions) else FastOptions())

        self._patch_locate()
        self.locate = None
        self._locate_enabled = locate is not False and self.fast is None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _patch_locate() -> None:
        _patch_locate_lora()

    def reference_logits(self, chunks):
        from .backend import quant_cache_scope

        if not self.prefill_opt and self.prefill_v2 in {"", "0", "off", "false", "none"} and os.environ.get("DECIDER_NVFP4_SHARED_QUANT", "0") == "1":
            with quant_cache_scope():
                return super().reference_logits(chunks)
        return super().reference_logits(chunks)

    def runtime_report(self) -> dict[str, Any]:
        modules = [m for m in self.backbone.modules() if isinstance(m, NVFP4Linear)]
        kept = [
            name
            for name, module in self.backbone.language_model.named_modules()
            if isinstance(module, nn.Linear) and not isinstance(module, NVFP4Linear)
        ]
        return {
            "precision": "nvfp4",
            "backend": backend_report(self.device),
            "checkpoint": self.nvfp4_checkpoint.summary(),
            "load": self.load_report,
            "nvfp4_modules": len(modules),
            "kept_bf16_linears": kept,
            "fast": self.fast is not None,
            "gdn_backend": self.gdn_backend,
            "prefill_v2": self.prefill_v2,
            "prefill_opt": self.prefill_opt,
            "deltanet_opt": self.deltanet_opt,
            "mode": self.mode,
            "locate_enabled": self._locate_enabled,
        }
