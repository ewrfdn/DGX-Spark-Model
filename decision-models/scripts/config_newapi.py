#!/usr/bin/env python3
"""把五个决策模型注册进 new-api（渠道 + 倍率）。

用法：
    export NEWAPI_TOKEN='<管理员 access token>'
    python3 config_newapi.py --dry-run     # 只打印将要做什么
    python3 config_newapi.py --apply       # 真正写入

价格基准（$/1M input tokens，输出免费）→ new-api 倍率 = 单价 / 2：
    bison-1                  0.10  -> 0.05
    elephant-1               0.035 -> 0.0175
    pplx-decider-v1.3        0.02  -> 0.01
    pplx-decider-flash-v1.3  0.01  -> 0.005
    GLiDE-0.4B               0.04  -> 0.02
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("NEWAPI_BASE", "http://127.0.0.1:3000")
TOKEN = os.environ.get("NEWAPI_TOKEN", "")
HOST = os.environ.get("DECISION_HOST", "192.168.101.2")   # 模型服务所在机器
GROUP = os.environ.get("NEWAPI_GROUP", "default")

# name, model id, port, price per 1M input tokens
MODELS = [
    ("decision-bison", "bison-1", 8011, 0.10),
    ("decision-elephant", "elephant-1", 8012, 0.035),
    ("decision-decider", "pplx-decider-v1.3", 8013, 0.02),
    ("decision-decider-flash", "pplx-decider-flash-v1.3", 8014, 0.01),
    ("decision-glide", "GLiDE-0.4B", 8015, 0.04),
]


def call(method: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("New-API-User", "1")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        return e.code, body


def list_channels() -> list[dict]:
    st, d = call("GET", "/api/channel/?p=0&page_size=100")
    if st != 200 or not isinstance(d, dict):
        raise SystemExit(f"列渠道失败 {st}: {d}")
    data = d.get("data")
    return data.get("items", []) if isinstance(data, dict) else (data or [])


def get_options() -> dict[str, str]:
    st, d = call("GET", "/api/option/")
    if st != 200 or not isinstance(d, dict):
        raise SystemExit(f"读配置失败 {st}: {d}")
    return {o["key"]: o.get("value", "") for o in d.get("data", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()
    if not TOKEN:
        raise SystemExit("请先 export NEWAPI_TOKEN=<管理员 access token>")

    existing = {c.get("name"): c for c in list_channels()}
    opts = get_options()
    ratio = json.loads(opts.get("ModelRatio") or "{}")
    comp = json.loads(opts.get("CompletionRatio") or "{}")

    print(f"new-api: {BASE}  现有渠道 {len(existing)} 个，现有模型 {len(ratio)} 个")
    actions = []

    for name, model, port, price in MODELS:
        base_url = f"http://{HOST}:{port}"
        want = {
            "name": name,
            "type": 1,                       # OpenAI 兼容
            "key": "sk-local-decision",
            "base_url": base_url,
            "models": model,
            "group": GROUP,
            "priority": 10,
            "weight": 10,
            "status": 2,                     # 先禁用，等容器起来再开
            "model_mapping": "",
        }
        if name in existing:
            cid = existing[name]["id"]
            print(f"= 更新渠道 {cid} {name} -> {base_url} / {model}")
            actions.append(("PUT", "/api/channel/", {**want, "id": cid}))
        else:
            print(f"+ 新建渠道 {name} -> {base_url} / {model}（status=2 禁用）")
            actions.append(("POST", "/api/channel/", {"mode": "single", "channel": want}))
        r = round(price / 2, 4)
        ratio[model] = r
        comp[model] = 0                      # 输出免费
        print(f"    ratio {model} = {r}  (${price}/1M input)")

    new_ratio = json.dumps(ratio, separators=(",", ":"))
    new_comp = json.dumps(comp, separators=(",", ":"))
    if new_ratio != (opts.get("ModelRatio") or ""):
        actions.append(("PUT", "/api/option/", {"key": "ModelRatio", "value": new_ratio}))
    if new_comp != (opts.get("CompletionRatio") or ""):
        actions.append(("PUT", "/api/option/", {"key": "CompletionRatio", "value": new_comp}))

    if not args.apply:
        print("\n[dry-run] 将执行：")
        for m, p, pl in actions:
            print(f"  {m} {p} {json.dumps(pl, ensure_ascii=False)[:160]}")
        print("\n加 --apply 生效")
        return

    for m, p, pl in actions:
        st, d = call(m, p, pl)
        ok = isinstance(d, dict) and d.get("success")
        print(f"{m} {p} -> {st} {'OK' if ok else d}")
        if not ok:
            sys.exit(1)
    print("\n完成。容器起来后把渠道 status 改成 1（启用）即可。")


if __name__ == "__main__":
    main()
