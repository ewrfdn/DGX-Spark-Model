"""Install the Decider's opt-in prefill optimizations on one backbone."""

from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps

from .backend import quant_cache_scope
from .deltanet_conv import install_deltanet_conv
from .deltanet_norm import install_deltanet_norm
from .plain_rmsnorm import install_plain_rmsnorm


def _cached_forward(original):
    @wraps(original)
    def forward(*args, **kwargs):
        with quant_cache_scope(enabled=True):
            return original(*args, **kwargs)

    return forward


def install_prefill_optimizations(backbone) -> Callable[[], None]:
    """Install the tested kernels and a per-decoder-layer activation cache."""
    with ExitStack() as installed:
        installed.callback(install_deltanet_norm(backbone))
        installed.callback(install_deltanet_conv(backbone))
        installed.callback(install_plain_rmsnorm(backbone, triton_reduction=True))

        for layer in backbone.language_model.layers:
            original = layer.forward
            if "forward" in vars(layer):
                installed.callback(setattr, layer, "forward", original)
            else:
                installed.callback(delattr, layer, "forward")
            layer.forward = _cached_forward(original)

        return installed.pop_all().close
