#!/usr/bin/env bash
# Pull a Docker Hub image WITHOUT letting the local Docker daemon do the network I/O.
#
# Why: on this host /etc/systemd/system/docker.service.d/proxy.conf points the Docker daemon at
# http://127.0.0.1:7890, whose upstream nodes cannot resolve (mihomo "dns resolve failed"), so every
# `docker pull` dies with `Get "https://<registry>/v2/": EOF` even though curl reaches the same
# registry fine. We are in the `docker` group but have no root, so the daemon config cannot be
# changed here. Instead: fetch manifest + config + blobs with curl (optionally through a fast
# domestic Docker Hub mirror), assemble a Docker "save" tar, and `docker load` it. dockerd then only
# reads a local file, and no registry traffic passes through its broken proxy.
#
# Usage:
#   ./pull-image.sh vllm/vllm-openai:qwen38-flash-next [--mirror docker.1ms.run] [--save-to FILE]
#
#   --mirror M   Docker Hub mirror to fetch through (default: none = registry-1.docker.io)
#   --save-to F  also keep the assembled tar at F (to copy to the other Spark and `docker load`)
#   --jobs N     parallel blob downloads (default 4)
set -euo pipefail

IMAGE_REF="${1:?usage: pull-image.sh REPO:TAG [--mirror M] [--save-to F] [--jobs N]}"; shift
MIRROR=""; SAVE_TO=""; JOBS=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mirror)  MIRROR="$2"; shift 2 ;;
    --save-to) SAVE_TO="$2"; shift 2 ;;
    --jobs)    JOBS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="${IMAGE_REF%:*}"; TAG="${IMAGE_REF##*:}"
[[ "$REPO" == */* ]] || REPO="library/$REPO"
if [[ "${REPO#*/}" == */* ]]; then HUB_NAME="${REPO#*/}"; else HUB_NAME="$REPO"; fi
SCOPE="repository:${HUB_NAME}:pull"

CURL=(curl -sSL --noproxy '*' --retry 5 --retry-all-errors --retry-delay 2 --max-time 900)

if [[ -n "$MIRROR" ]]; then
  REG="$MIRROR"
  # Learn the auth endpoint from the registry's own 401 challenge: mirrors do not all serve a
  # token endpoint under their own host (docker.m.daocloud.io points at m.daocloud.io/auth/token).
  CHALLENGE=$(curl -sIS --noproxy '*' --max-time 30 "https://$MIRROR/v2/" 2>/dev/null | tr -d '\r' |
              sed -n 's/^[Ww][Ww][Ww]-[Aa]uthenticate: *[Bb]earer *//p' | head -1)
  REALM=$(sed -n 's/.*realm="\([^"]*\)".*/\1/p' <<< "$CHALLENGE")
  SERVICE=$(sed -n 's/.*service="\([^"]*\)".*/\1/p' <<< "$CHALLENGE")
  if [[ -n "$REALM" ]]; then
    TOKEN=$("${CURL[@]}" "${REALM}?service=${SERVICE:-$MIRROR}&scope=$SCOPE" |
            python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("access_token") or d.get("token") or "")')
  else
    TOKEN=$("${CURL[@]}" "https://$MIRROR/openapi/v1/auth/token?service=$MIRROR&scope=$SCOPE" |
            python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("access_token") or d.get("token") or "")')
  fi
else
  REG="registry-1.docker.io"
  TOKEN=$("${CURL[@]}" "https://auth.docker.io/token?service=registry.docker.io&scope=$SCOPE" |
          python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
fi
[[ -n "$TOKEN" ]] || { echo "ERROR: no registry token for $REPO" >&2; exit 1; }

# Reusable work dir keyed by image ref: a failed or interrupted pull resumes next time instead of
# starting over (on this link a 9 GiB image is a multi-hour download).
WORK="${TMPDIR:-/tmp}/imgpull-$(printf '%s' "$IMAGE_REF" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p "$WORK"
if [[ -z "${KEEP_PULL_DIR:-}" ]]; then
  trap 'case "${PULL_OK:-0}" in 1) rm -rf -- "$WORK" ;; *) echo "[pull-image] partial data kept in $WORK (rerun to resume)" >&2 ;; esac' EXIT
fi
cd "$WORK"

ACCEPT='application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'
echo "[pull-image] $IMAGE_REF from $REG"
"${CURL[@]}" -H "Authorization: Bearer $TOKEN" -H "Accept: $ACCEPT" \
  "https://$REG/v2/$REPO/manifests/$TAG" -o manifest.json

DIGEST=$(python3 - *<<'PY'
import json, sys, platform
arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine(), platform.machine())
d = json.load(open("manifest.json"))
if "manifests" not in d:
    print(""); raise SystemExit
for m in d["manifests"]:
    p = m.get("platform") or {}
    if p.get("architecture") == arch and p.get("os") == "linux":
        print(m["digest"]); raise SystemExit
raise SystemExit("no manifest for linux/%s" % arch)
PY
)
if [[ -n "$DIGEST" ]]; then
  echo "[pull-image] arch manifest $DIGEST"
  "${CURL[@]}" -H "Authorization: Bearer $TOKEN" -H "Accept: $ACCEPT" \
    "https://$REG/v2/$REPO/manifests/$DIGEST" -o image-manifest.json
else
  cp manifest.json image-manifest.json
fi

CONFIG_SHA=$(python3 -c 'import json; print(json.load(open("image-manifest.json"))["config"]["digest"].split(":")[1])')
echo "[pull-image] config ${CONFIG_SHA:0:12}"
"${CURL[@]}" -H "Authorization: Bearer $TOKEN" "https://$REG/v2/$REPO/blobs/sha256:$CONFIG_SHA" -o config.json

python3 - <<'PY' > layers.txt
import json
for l in json.load(open("image-manifest.json"))["layers"]:
    print(l["digest"].split(":")[1], l["size"])
PY

# Each blob: fetch with resume until its size matches, then move into place.
# (Numbers are positional params, not `local a=$1`: under `set -u` bash expands $1 before `local`
#  takes effect, and this function is exported and run from a subshell.)
#
# Only `--fail`'d, range-honouring responses are appended: a resumed request MUST answer 206.
# If the CDN ignores Range and answers 200 (the whole blob again), or the transfer dies early,
# the chunk is thrown away and retried from scratch — appending it blindly is what once grew a
# layer past its declared size.
fetch_layer() {
  local f="blob-$1.part" chunk attempt have rc
  for attempt in $(seq 1 60); do
    have=0; [[ -f $f ]] && have=$(stat -c%s "$f")
    if (( have == $2 )); then
      mv -f "$f" "blob-$1"
      echo "[pull-image]   done ${1:0:12} $(awk -v s=$2 'BEGIN{printf "%.0f MiB", s/1048576}')" >&2
      return 0
    fi
    (( have > $2 )) && { rm -f "$f"; have=0; }        # never resume past the end
    chunk="$f.chunk"
    : > "$chunk"
    # --speed-limit/--speed-time: drop a stalled chunk quickly instead of waiting out --max-time
    # -L: registries answer a blob GET with a 307 redirect to their CDN
    curl -fSL --noproxy '*' --max-time 300 --speed-limit 10000 --speed-time 30 \
      -H "Authorization: Bearer $TOKEN" -H "Range: bytes=${have}-" \
      "https://$REG/v2/$REPO/blobs/sha256:$1" >> "$chunk" 2>/dev/null
    rc=$?
    if (( rc == 0 )); then
      cat "$chunk" >> "$f"                            # complete tail: append
    elif (( rc == 33 )); then
      :                                               # server ignored Range; nothing safe to use
    else
      # truncated/failed transfer: keep what arrived, it starts exactly at $have
      [[ -s $chunk ]] && cat "$chunk" >> "$f"
    fi
    rm -f "$chunk"
  done
  have=0; [[ -f $f ]] && have=$(stat -c%s "$f")
  if (( have == $2 )); then mv -f "$f" "blob-$1"; else echo "ERROR: layer ${1:0:12} $have/$2" >&2; return 1; fi
}
export -f fetch_layer
export TOKEN REG REPO

echo "[pull-image] $(wc -l < layers.txt) layers, $(( $(awk '{s+=$2} END{print s}' layers.txt) / 1048576 )) MiB"
running=0
while read -r sha size; do
  ( fetch_layer "$sha" "$size" ) &
  running=$((running + 1))
  if (( running >= JOBS )); then wait -n || true; running=$((running - 1)); fi
done < layers.txt
wait
# every layer must be present at its declared size...
missing=0
while read -r sha size; do
  [[ -f "blob-$sha" ]] || { echo "ERROR: missing layer $sha" >&2; missing=1; }
done < layers.txt
(( missing == 0 )) || exit 1
# ...and hash to its digest, so a silently corrupted resume cannot produce a broken image
echo "[pull-image] verifying layer digests (sha256)"
while read -r sha size; do
  got=$(sha256sum "blob-$sha" | cut -d' ' -f1)
  if [[ "$got" != "$sha" ]]; then
    echo "ERROR: layer ${sha:0:12} digest mismatch (got ${got:0:12}); removing it — rerun to refetch" >&2
    rm -f "blob-$sha"; missing=1
  fi
done < layers.txt
(( missing == 0 )) || exit 1

IMAGE_REF="$IMAGE_REF" python3 - <<'PY'
import io, json, os, tarfile
man = json.load(open("image-manifest.json"))
cfg = man["config"]["digest"].split(":")[1]
with tarfile.open("image.tar", "w") as t:
    t.add("config.json", arcname=cfg + ".json")
    for l in man["layers"]:
        sha = l["digest"].split(":")[1]
        t.add("blob-" + sha, arcname=sha + "/layer.tar")
    data = json.dumps([{
        "Config": cfg + ".json",
        "RepoTags": [os.environ["IMAGE_REF"]],
        "Layers": [l["digest"].split(":")[1] + "/layer.tar" for l in man["layers"]],
    }]).encode()
    ti = tarfile.TarInfo("manifest.json"); ti.size = len(data)
    t.addfile(ti, io.BytesIO(data))
print("[pull-image] image.tar %.2f GiB" % (os.path.getsize("image.tar") / 2**30))
PY

docker load -i image.tar
docker image inspect "$IMAGE_REF" --format '{{.Id}} {{.RepoTags}}' >/dev/null
echo "[pull-image] loaded $IMAGE_REF"
PULL_OK=1
[[ -z "$SAVE_TO" ]] || { cp image.tar "$SAVE_TO"; echo "[pull-image] saved tar -> $SAVE_TO"; }
