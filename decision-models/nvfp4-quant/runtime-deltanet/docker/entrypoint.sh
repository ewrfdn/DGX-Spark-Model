#!/usr/bin/env bash
set -euo pipefail

# Model checkpoints carry their runtime as source, not as a wheel:
#   <model>/source/src  -> pplx_decider   (Perplexity)
#   <model>/glide       -> glide          (Fastino)
# Put every such path on PYTHONPATH so the gateway can import on demand.
EXTRA=""
if [ -d "${MODEL_ROOT:-/models}" ]; then
  while IFS= read -r d; do
    EXTRA="${EXTRA:+$EXTRA:}$d"
  done < <(find "${MODEL_ROOT:-/models}" -maxdepth 3 -type d \( -name src -path '*/source/src' -o -name glide \) 2>/dev/null | sort -u)
fi

export PYTHONPATH="${PYTHONPATH:-}${EXTRA:+:$EXTRA}"
echo "[entrypoint] MODEL_ROOT=${MODEL_ROOT:-/models} DEVICE=${DEVICE:-cuda:0} MAX_LOADED=${MAX_LOADED:-1}"
echo "[entrypoint] PYTHONPATH=${PYTHONPATH}"
ls -1 "${MODEL_ROOT:-/models}" 2>/dev/null | sed 's/^/[entrypoint] model: /' || true

exec "$@"
