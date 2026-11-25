# ---------------------------------------------------------------------------------------
# W4A4 NVFP4 inference container for pplx-decider-flash-v1.3 on NVIDIA GB10 (SM121)
#
#   base  : NGC PyTorch 26.07  -> torch 2.13.0a0+cu13.3, CUDA 13.3, aarch64
#   quant : FlashInfer 0.6.18  -> nvfp4_quantize(backend='cuda') + mm_fp4(backend='cutlass' by default; b12x optional)
#   model : transformers 5.17.0 (the revision the checkpoint ships its code for)
#
# Build (from the runtime-deltanet directory, after ./docker/fetch_wheels.sh):
#   docker build -f docker/Dockerfile -t decision-nvfp4:0.6.18 .
#
# Run:
#   docker run --gpus all --ipc=host -p 8016:8016 \
#     -v /home/sakana/models/decision-models:/models:ro \
#     -v nvfp4-fi-cache:/opt/fi-cache \
#     -e MODEL=pplx-decider-flash-v1.3-NVFP4 \
#     decision-nvfp4:0.6.18
# ---------------------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FLASHINFER_WORKSPACE_BASE=/opt/fi-cache \
    FLASHINFER_CACHE_DIR=/opt/fi-cache

COPY docker/wheels/ /opt/wheels/

# FlashInfer 0.6.18 (no-deps: torch / numpy / tvm_ffi / ninja / cuda-python /
# cutlass-dsl 4.5.2 all come from the NGC base).
RUN python -m pip install --no-index --no-deps /opt/wheels/flashinfer_python-0.6.18-py3-none-any.whl \
 && python -m pip install --no-index --no-deps \
        /opt/wheels/transformers-5.17.0-py3-none-any.whl \
        /opt/wheels/tokenizers-0.23.2-*.whl \
        /opt/wheels/huggingface_hub-1.33.0-py3-none-any.whl \
        /opt/wheels/flash_linear_attention-0.6.0-py3-none-any.whl \
        /opt/wheels/fla_core-0.6.0-py3-none-any.whl \
 && python -m pip install --no-index --find-links /opt/wheels \
        "fastapi==0.142.2" "uvicorn[standard]==0.54.0" \
 && python - <<'PY'
import flashinfer, torch, transformers
print("flashinfer", flashinfer.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
PY

# FlashInfer JIT/cuTe-DSL cache lives in a named volume so the (one-time)
# quantizer + cutlass compile survives container restarts.  See docker/warm_cache.sh.
RUN mkdir -p /opt/fi-cache && chmod 777 /opt/fi-cache

WORKDIR /app
COPY nvfp4/ /app/nvfp4/
COPY docker/warmup_cache.py /app/scripts/warmup_cache.py
COPY serving/app/ /app/app/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV MODEL_ROOT=/models \
    DEVICE=cuda:0 \
    MAX_LOADED=1 \
    DECIDER_PRECISION=nvfp4 \
    DECIDER_NVFP4_BACKEND=cutlass \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    DECIDER_NVFP4_PDL=0 \
    DECIDER_PREFILL_V2=full \
    PPLX_MODE=isolated \
    PPLX_FAST=0 \
    PYTHONPATH=/app

EXPOSE 8016
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8016"]
