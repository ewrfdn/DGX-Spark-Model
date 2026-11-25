# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------------------
# NVFP4 离线权重量化镜像（与生产推理环境完全隔离）
#
#   base : NGC PyTorch 26.07
#          -> torch 2.13.0a0+cu13.3 / nvcc 13.3 / CUDA 13.3 / aarch64（GB10 可用）
#   tool : FlashInfer 0.6.3（flashinfer.fp4_quantization.nvfp4_quantize）
#
# 依赖策略：base 镜像已自带 numpy 2.1.0 / ninja / tvm_ffi 0.1.12 / safetensors 0.8.0，
# 因此只装 flashinfer 本体（--no-deps），刻意不引入 cutlass-dsl / cudnn-frontend 等
# 与本任务无关的重量级依赖，也不引入任何其它量化框架。
#
# 构建：
#   ./fetch_wheels.sh                 # 下载 wheels/flashinfer_python-*.whl
#   docker build -t nvfp4-quant:0.6.3 .
# ---------------------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG FLASHINFER_VERSION=0.6.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    FLASHINFER_CACHE_DIR=/opt/fi-cache

COPY wheels/ /opt/wheels/

RUN python -m pip install --no-index --no-deps /opt/wheels/flashinfer_python-*.whl \
    && python - <<'PY'
import inspect, torch, flashinfer
from flashinfer.fp4_quantization import nvfp4_quantize, SfLayout
print("flashinfer", flashinfer.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("nvfp4_quantize", inspect.signature(nvfp4_quantize))
print("SfLayout", {m.name: int(m.value) for m in SfLayout})
PY

RUN mkdir -p /opt/fi-cache /work
WORKDIR /work
