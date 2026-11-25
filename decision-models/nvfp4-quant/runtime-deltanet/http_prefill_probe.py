"""Check the deployed NVFP4 API and benchmark fixed synthetic text workloads."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time
from urllib.request import Request, urlopen


MODEL = "pplx-decider-flash-v1.3-NVFP4"
PARAGRAPH = (
    "The customer reported that checkout failed after rotating an API key "
    "and that a duplicate charge is still pending on their corporate card. "
)
QUESTIONS = {
    "escalate": {"type": "noul", "instructions": "Does this incident need a human reviewer?"},
    "route": {
        "type": "choice", "instructions": "Which team needs to handle the incident?",
        "criteria": {"billing": "Duplicate charge", "engineering": "Product outage", "sales": "New account"},
    },
    "urgent": {"type": "noul", "instructions": "Is the incident urgent?"},
}


def request_json(url, payload=None):
    if payload is not None:
        url = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(url, timeout=120) as result:
        return json.load(result), 1000 * (time.perf_counter() - started)


def decision_payload(repeats, count):
    questions = dict(list(QUESTIONS.items())[:count])
    return {"model": MODEL, "state": PARAGRAPH * repeats, "questions": questions}


def check_answers(answers, questions):
    probability = answers["escalate"]["noul"]
    if answers.keys() != questions.keys() or not (math.isfinite(probability) and 0 <= probability <= 1):
        raise RuntimeError("Decision endpoint returned unexpected answers")


def check_decision(result, payload):
    check_answers(result["answers"], payload["questions"])
    usage = result["usage"]
    if result["model"] != MODEL or usage["input_tokens"] <= 0 or usage["latency_ms"] <= 0:
        raise RuntimeError("Decision endpoint returned an unexpected model or token/latency usage")


def smoke(base_url):
    status, _ = request_json(f"{base_url}/health")
    if not status["ok"] or MODEL not in status["loaded"]:
        raise RuntimeError(f"Preload check failed: {status}")
    models, _ = request_json(f"{base_url}/v1/models")
    if MODEL not in {item["id"] for item in models["data"]}:
        raise RuntimeError("Model is missing from /v1/models")
    payload = decision_payload(1, 3)
    result, _ = request_json(f"{base_url}/v1/systemone", payload)
    check_decision(result, payload)
    chat, _ = request_json(f"{base_url}/v1/chat/completions", payload)
    check_answers(json.loads(chat["choices"][0]["message"]["content"]), payload["questions"])
    if chat["model"] != MODEL or chat["usage"]["input_tokens"] != result["usage"]["input_tokens"]:
        raise RuntimeError("Chat facade returned different model or token usage")
    return {"model_loaded": MODEL, "systemone_usage": result["usage"], "chat_usage": chat["usage"]}


def benchmark(base_url, repeat):
    rows = []
    for target, repeats in ((2000, 75), (8000, 300)):
        for count in (1, 3):
            payload = decision_payload(repeats, count)
            start, _ = request_json(f"{base_url}/v1/systemone", payload)
            check_decision(start, payload)
            samples = []
            for _ in range(repeat):
                result, wall_ms = request_json(f"{base_url}/v1/systemone", payload)
                check_decision(result, payload)
                if start["answers"] != result["answers"] or start["usage"]["input_tokens"] != result["usage"]["input_tokens"]:
                    raise RuntimeError("Repeated requests returned different answers or token usage")
                samples.append({"wall_ms": wall_ms, "server_ms": result["usage"]["latency_ms"]})
            tokens = start["usage"]["input_tokens"]
            server_p50 = median(sample["server_ms"] for sample in samples)
            row = {
                "target_tokens": target, "questions": count, "input_tokens": tokens,
                "server_p50_ms": server_p50, "server_tokens_per_second": 1000 * tokens / server_p50,
                "wall_p50_ms": median(sample["wall_ms"] for sample in samples),
                "warmup_server_ms": start["usage"]["latency_ms"], "samples": samples,
                "answers": start["answers"],
            }
            print(f"{tokens} tokens / {count} questions: {server_p50:.2f} ms", flush=True)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8016")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3, help="Use 0 to run only API smoke checks")
    args = parser.parse_args()
    if args.repeat < 0:
        parser.error("--repeat must be nonnegative")
    base_url = args.url.rstrip("/")
    smoke_report = smoke(base_url)
    buckets = benchmark(base_url, args.repeat) if args.repeat else []
    report = {"url": base_url, "repeat": args.repeat, "smoke": smoke_report, "buckets": buckets}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
