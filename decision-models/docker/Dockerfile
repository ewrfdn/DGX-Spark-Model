# One container, three decision-model runtimes, one System One API.
#
#   * qev (OneJev @81ce62f + Base Labs serve.patch)  -> elephant-1, bison-1
#   * pplx_decider (shipped in the Perplexity repo)  -> decider / decider flash
#   * glide (shipped in the Fastino repo)            -> GLiDE-0.4B
#
# Build:  docker build -t decision-gateway ./serving
# Run:    docker run --gpus all -p 8000:8000 -v $PWD:/models decision-gateway
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pinned exactly like the model cards: torch 2.14 / transformers 5.17 (CUDA 13 wheels).
RUN pip install --upgrade pip \
 && pip install torch==2.14.0 torchvision==0.29.0 \
 && pip install transformers==5.17.0 tokenizers==0.23.2 safetensors==0.8.0 \
      "numpy<3" pillow==12.3.0 jinja2==3.1.6 accelerate "huggingface_hub>=1.31" \
      fastapi==0.142.2 "uvicorn[standard]==0.54.0" pydantic

# OneJev qev runtime, pinned to the commit Base Labs served with, plus their readout patch.
ARG ONEJEV_REPO=https://github.com/OmniJev/OneJev.git
ARG ONEJEV_COMMIT=81ce62f1597c91e46767d4d02d6ac2e18534fe94
COPY serve.patch /tmp/serve.patch
RUN git clone --filter=blob:none "$ONEJEV_REPO" /opt/onejev \
 && cd /opt/onejev \
 && git checkout "$ONEJEV_COMMIT" \
 && git apply --verbose /tmp/serve.patch \
 && pip install -e .

WORKDIR /app
COPY app/ /app/app/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV MODEL_ROOT=/models \
    DEVICE=cuda:0 \
    MAX_LOADED=1 \
    PYTHONPATH=/opt/onejev:/app

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
