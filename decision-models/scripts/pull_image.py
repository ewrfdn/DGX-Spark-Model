#!/usr/bin/env python3
"""Pull an image without the Docker daemon and write a `docker load`-able tar.

The daemon on this box is pinned to a dead HTTP proxy, and its NO_PROXY list does not
cover the auth or blob CDN hosts, so `docker pull` always ends in EOF. Doing the registry
dance ourselves (with a direct connection) and handing the result to `docker load` works.

    python3 pull_image.py python:3.12-slim -o python312.tar [--arch arm64]
    docker load -i python312.tar
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

MIRROR = os.environ.get("PULL_MIRROR", "docker.1ms.run")   # set to "" for Docker Hub direct
ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


UA = "docker/28.3.3 (linux; arm64)"


def open_url(url, headers=None, data=None, method=None):
    hdrs = {"User-Agent": UA}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs, data=data, method=method)
    return urllib.request.urlopen(req, timeout=120)


def split_ref(ref: str):
    if "/" in ref.split(":")[0] or ref.count("/") >= 2:
        registry, rest = ref.split("/", 1)
    else:
        registry, rest = "registry-1.docker.io", ref
    if ":" in rest:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, "latest"
    if "/" not in repo:
        repo = "library/" + repo
    return registry, repo, tag


def token_for(registry: str, repo: str) -> str:
    try:
        open_url(f"https://{registry}/v2/")
        hdr = ""
    except urllib.error.HTTPError as e:
        hdr = e.headers.get("WWW-Authenticate", "")
    realm = re.search(r'realm="([^"]+)"', hdr)
    service = re.search(r'service="([^"]+)"', hdr)
    if not realm:
        return ""
    url = f"{realm.group(1)}?scope=repository:{repo}:pull"
    if service:
        url += f"&service={service.group(1)}"
    with open_url(url) as r:
        return json.load(r).get("token") or json.loads("{}").get("access_token", "")


def fetch_json(url, headers):
    with open_url(url, headers=headers) as r:
        return json.load(r), r.headers.get("Content-Type", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--arch", default="arm64")
    args = ap.parse_args()

    registry, repo, tag = split_ref(args.ref)
    if MIRROR:
        registry = MIRROR
    api_repo = repo
    base = f"https://{registry}/v2/{api_repo}"
    print(f"registry={registry} repo={api_repo} tag={tag} arch={args.arch}", flush=True)

    tok = token_for(registry, api_repo)
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    headers["Accept"] = ACCEPT

    doc, _ = fetch_json(f"{base}/manifests/{tag}", headers)
    if "manifests" in doc:
        pick = None
        for m in doc["manifests"]:
            p = m.get("platform", {})
            if p.get("architecture") == args.arch and p.get("os", "linux") == "linux":
                pick = m
                break
        if pick is None:
            sys.exit(f"no {args.arch} manifest in {args.ref}")
        print(f"  selected {pick['digest'][:24]}", flush=True)
        doc, _ = fetch_json(f"{base}/manifests/{pick['digest']}", headers)

    config = doc["config"]
    layers = doc["layers"]
    total = config["size"] + sum(l["size"] for l in layers)
    print(f"  config {config['size']/1e6:.2f} MB + {len(layers)} layers "
          f"{sum(l['size'] for l in layers)/1e6:.1f} MB", flush=True)

    def blob(digest: str, expected: int) -> bytes:
        url = f"{base}/blobs/{digest}"
        with open_url(url, headers=headers) as r:
            data = r.read()
        if len(data) != expected:
            raise RuntimeError(f"{digest}: got {len(data)} want {expected}")
        return data

    cfg_bytes = blob(config["digest"], config["size"])
    cfg_digest = hashlib.sha256(cfg_bytes).hexdigest()

    out = tarfile.open(args.output, "w")
    entries = []
    for i, layer in enumerate(layers):
        print(f"  layer {i+1}/{len(layers)} {layer['size']/1e6:.1f} MB", flush=True)
        raw = blob(layer["digest"], layer["size"])
        if layer.get("mediaType", "").endswith("gzip") or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        name = f"l{i}"
        info = tarfile.TarInfo(f"{name}/layer.tar")
        info.size = len(raw)
        out.addfile(info, io.BytesIO(raw))
        v = b"1.0"
        vi = tarfile.TarInfo(f"{name}/VERSION")
        vi.size = len(v)
        out.addfile(vi, io.BytesIO(v))
        entries.append(f"{name}/layer.tar")

    cfg_name = f"{cfg_digest}.json"
    info = tarfile.TarInfo(cfg_name)
    info.size = len(cfg_bytes)
    out.addfile(info, io.BytesIO(cfg_bytes))

    manifest = json.dumps([{
        "Config": cfg_name,
        "RepoTags": [args.ref],
        "Layers": entries,
    }]).encode()
    mi = tarfile.TarInfo("manifest.json")
    mi.size = len(manifest)
    out.addfile(mi, io.BytesIO(manifest))
    out.close()
    print(f"wrote {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
