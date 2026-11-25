#!/usr/bin/env python3
"""Verify downloaded repos against the HF API (size check).

Usage: python3 verify.py [repo_id ...]   (default: all repos under ROOT)
"""
import json, os, sys, urllib.request

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
ROOT = os.environ.get("MODEL_ROOT", os.path.dirname(os.path.abspath(__file__)))

REPOS = {
    "GLiDE-0.4B": "fastino/GLiDE-0.4B",
    "elephant-1": "baselabs/elephant-1",
    "bison-1": "baselabs/bison-1",
    "pplx-decider-v1.3": "perplexity-ai/pplx-decider-v1.3",
    "pplx-decider-flash-v1.3": "perplexity-ai/pplx-decider-flash-v1.3",
}


def api(path):
    req = urllib.request.Request(f"{MIRROR}/api/{path}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def check(name, repo):
    dest = os.path.join(ROOT, name)
    if not os.path.isdir(dest):
        print(f"{name:26} MISSING"); return False
    tree = api(f"models/{repo}/tree/main?recursive=1")
    sizes = {e["path"]: e.get("size", 0) for e in tree if e.get("type") == "file"}
    ok = True
    total = 0
    for p, want in sizes.items():
        if p == ".gitattributes":
            continue
        f = os.path.join(dest, p)
        if not os.path.exists(f):
            print(f"  MISS {p}"); ok = False; continue
        got = os.path.getsize(f)
        total += got
        if want and got != want:
            print(f"  SIZE {p}: {got} != {want}"); ok = False
    print(f"{name:26} {'OK' if ok else 'INCOMPLETE':10} {total/1e9:6.2f} GB / {sum(sizes.values())/1e9:6.2f} GB")
    return ok


if __name__ == "__main__":
    names = sys.argv[1:] or list(REPOS)
    allok = all(check(n, REPOS[n]) for n in names if n in REPOS)
    sys.exit(0 if allok else 1)
