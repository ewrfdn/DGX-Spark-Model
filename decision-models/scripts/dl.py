#!/usr/bin/env python3
"""Download HF repos through hf-mirror using aria2c (multi-connection).

Usage: python3 dl.py <repo_id> [<repo_id> ...] --root /path/to/root [--jobs 3] [--conn 8]
Writes each repo into <root>/<name-without-org>/.
"""
import json, os, subprocess, sys, urllib.request, urllib.parse

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")


def api(path):
    req = urllib.request.Request(f"{MIRROR}/api/{path}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def build_list(repo, dest):
    info = api(f"models/{repo}")
    files = []
    for s in info["siblings"]:
        p = s["rfilename"]
        if p == ".gitattributes":
            continue
        files.append(p)
    # sizes
    tree = api(f"models/{repo}/tree/main?recursive=1")
    sizes = {e["path"]: e.get("size", 0) for e in tree if e.get("type") == "file"}
    total = sum(sizes.get(f, 0) for f in files)
    print(f"[{repo}] {len(files)} files, {total/1e9:.2f} GB -> {dest}", flush=True)
    lines = []
    for p in files:
        url = f"{MIRROR}/{repo}/resolve/main/{urllib.parse.quote(p)}"
        out = p.replace("/", "%2F") if False else p
        lines.append(url)
        lines.append(f"  dir={dest}")
        lines.append(f"  out={out}")
        os.makedirs(os.path.dirname(os.path.join(dest, p)), exist_ok=True)
    return lines, total


def main():
    args = sys.argv[1:]
    root = "/home/sakana/models/decision-models"
    jobs, conn = 3, 8
    repos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--root":
            root = args[i + 1]; i += 2
        elif a == "--jobs":
            jobs = int(args[i + 1]); i += 2
        elif a == "--conn":
            conn = int(args[i + 1]); i += 2
        else:
            repos.append(a); i += 1

    os.makedirs(root, exist_ok=True)
    all_lines = []
    for repo in repos:
        dest = os.path.join(root, repo.split("/")[-1])
        os.makedirs(dest, exist_ok=True)
        lines, _ = build_list(repo, dest)
        all_lines += lines
    inp = os.path.join(root, ".aria2_input.txt")
    with open(inp, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    cmd = [
        "aria2c", f"--input-file={inp}", f"--max-concurrent-downloads={jobs}",
        f"--max-connection-per-server={conn}", "--split=8", "--min-split-size=8M",
        "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=false",
        "--file-allocation=none", "--console-log-level=warn", "--summary-interval=30",
        "--connect-timeout=30", "--timeout=60", "--max-tries=20", "--retry-wait=5",
        "--lowest-speed-limit=0",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
