# DeepSeek-V4.1-Flash TensorFold native Zig engine image (2x DGX Spark / GB10 sm_121)
#
# Built on nvcr.io/nvidia/pytorch:26.07-py3 (CUDA 13.3, NCCL, cuBLAS, libibverbs, Pillow).
# Compiles BertholomusAI's TensorFold fork (branch deepseek-v41-zig @ TF_REF) with MiaAI-Lab's
# patch series (0001..0068: Spill v2, KV sharding, 0040 restore-room, 0041/0042/0045 chunked fill
# & cut-in, 0043 picture-resume-skip, 0059 lane-thread CUDA bind, 0033 graph-cap, 0066/0068, etc.),
# plus tf-dsv41-rank-cache and standalone input helpers from INPUTS_REF.
#
# P0 CPU optimization:
#   Builds natively for the host GB10 CPU (NO `-Dtarget=aarch64-linux-gnu.2.28` generic downgrade),
#   so ARMv9.2 hardware CRC32 + SHA256 instructions are enabled for Spill v2 block hashing/checks.
#   Pass ZIG_BUILD_EXTRA="-Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline+crc+sha2" only if you
#   explicitly need cross-glibc portability while preserving hardware CRC32/SHA256.
#
# Use scripts/build_image.sh to stage patches/ and patches-inputs/ into the build context and run
# `docker build` with deterministic PATCHES_HASH labels.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

# Optional tools needed by make_bias_vl.py / ABLIT preparation on CPU.
RUN python3 -m pip install --no-cache-dir safetensors

ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ARG ZIG_MIRRORS="https://zig.linus.dev/zig"
# Optional offline payload directory (always present in context; may contain pre-downloaded
# zig-aarch64-linux-${ZIG_VERSION}.tar.xz and/or bare git mirrors tensorfold.git).
COPY offline /opt/offline

RUN set -e; \
    tarball="/opt/offline/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"; \
    if [ -s "${tarball}" ]; then \
      echo "zig: using offline tarball ${tarball}"; \
      echo "${ZIG_SHA256}  ${tarball}" | sha256sum -c -; \
      cp "${tarball}" /tmp/zig.tar.xz; \
    else \
      ok=0; \
      for base in ${ZIG_MIRRORS} "https://ziglang.org/download/${ZIG_VERSION}"; do \
        url="${base%/}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"; \
        echo "zig: trying ${url}"; \
        if curl --connect-timeout 20 --max-time 300 --retry 3 --retry-delay 5 --retry-max-time 600 \
             -fsSL -o /tmp/zig.tar.xz "${url}" && \
           echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -; then \
          ok=1; break; \
        fi; \
        rm -f /tmp/zig.tar.xz; \
      done; \
      [ "${ok}" = 1 ] || { echo "zig ${ZIG_VERSION}: no source served pinned sha256" >&2; exit 1; }; \
    fi; \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm -f /tmp/zig.tar.xz && \
    /opt/zig/zig version

ARG SM=121
ARG TF_REPO=https://github.com/bertholomus/TensorFold.git
ARG TF_REF=fbbc97132cd004237d9887b8d3185896efaedc5c
ARG INPUTS_REF=59402086381a42c74d53053443224eec2790b4c5
ARG ZIG_BUILD_EXTRA=""

# ---- 1. Serving engine (tensorfold-native + tf-dsv41-lanes + CLI tools) ----
RUN set -e; \
    src="${TF_REPO}"; \
    if [ -d /opt/offline/tensorfold.git ]; then src="/opt/offline/tensorfold.git"; echo "tf: using offline repo ${src}"; fi; \
    git init -q /tensorfold && cd /tensorfold && git remote add origin "${src}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"

COPY patches /opt/tf-patches
RUN set -e; cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do \
      [ -e "$p" ] || continue; \
      echo "applying $p"; \
      git apply --check "$p" && git apply "$p" || exit 1; \
    done

# Native GB10 build (ReleaseFast, host ARMv9.2 CPU features enabled -> hardware CRC32/SHA256)
RUN set -e; cd /tensorfold && \
    /opt/zig/zig build \
      -Dnvcc=/usr/local/cuda/bin/nvcc \
      -Dsm=${SM} \
      -Doptimize=ReleaseFast \
      ${ZIG_BUILD_EXTRA} \
      --prefix /opt/tensorfold \
      -j"$(nproc)" \
      install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && \
    test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && \
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/; fi; \
    done && \
    if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then \
      cp -a tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md; \
    fi && \
    rm -rf /tensorfold /root/.cache/zig

# ---- 2. Inputs tools (tf-dsv41-rank-cache + make_token_map/make_rope/make_bias_vl) ----
RUN set -e; \
    src="${TF_REPO}"; \
    if [ -d /opt/offline/tensorfold.git ]; then src="/opt/offline/tensorfold.git"; fi; \
    git init -q /tf-inputs && cd /tf-inputs && git remote add origin "${src}" && \
    git fetch -q --depth 1 origin "${INPUTS_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${INPUTS_REF}"

COPY patches-inputs /opt/tf-inputs-patches
RUN set -e; cd /tf-inputs && \
    for p in /opt/tf-inputs-patches/*.patch; do \
      [ -e "$p" ] || continue; \
      echo "applying $p"; \
      git apply --check "$p" && git apply "$p" || exit 1; \
    done

RUN set -e; cd /tf-inputs && \
    /opt/zig/zig build \
      -Dnvcc=/usr/local/cuda/bin/nvcc \
      -Dsm=${SM} \
      -Doptimize=ReleaseFast \
      ${ZIG_BUILD_EXTRA} \
      --prefix /opt/inputs \
      -j"$(nproc)" \
      install && \
    test -x /opt/inputs/bin/tf-dsv41-rank-cache && \
    cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ && \
    mkdir -p /opt/tensorfold/tools/inputs && \
    cp tools/dsv41-zig/inputs/make_token_map.py \
       tools/dsv41-zig/kit/make_rope.py \
       tools/dsv41-zig/kit/make_bias_vl.py \
       /opt/tensorfold/tools/inputs/ && \
    if [ -e LICENSE ]; then cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-inputs-branch; fi && \
    rm -rf /tf-inputs /opt/inputs /opt/offline /root/.cache/zig

ARG PATCHES_HASH=unknown
ARG MIAAI_REF=c7f2b85100f0a7b099c92a4e21825e30d93c9c51
LABEL tf.patches=${PATCHES_HASH} \
      tf.engine=zig \
      tf.ref=${TF_REF} \
      tf.inputs=${INPUTS_REF} \
      tf.miaai=${MIAAI_REF}

ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
