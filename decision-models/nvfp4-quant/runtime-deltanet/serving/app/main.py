"""Unified System One gateway for decision models with different runtimes.

One process, one port, one wire format (/v1/systemone). Models are loaded lazily
and dispatched to the runtime each checkpoint actually ships:

  * qev  (OneJev + serve.patch)  -> baselabs/elephant-1, baselabs/bison-1   (readout.json)
  * pplx_decider                 -> perplexity-ai/pplx-decider-v1.3, -flash-v1.3
  * glide                        -> fastino/GLiDE-0.4B

Adapters are chosen by looking at the checkpoint directory, so adding a model is
just dropping it under MODEL_ROOT.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
MAX_LOADED = int(os.environ.get("MAX_LOADED", "1"))
ONLY_MODEL = os.environ.get("MODEL") or None  # serve a single model per container
PRELOAD = [m.strip() for m in os.environ.get("PRELOAD", "").split(",") if m.strip()]


# --------------------------------------------------------------------------- #
# request / response schema (TypeSafe System One compatible)
# --------------------------------------------------------------------------- #
class Question(BaseModel):
    type: str
    instructions: str | None = None
    criteria: Any = None


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any
    questions: dict[str, Question] = Field(default_factory=dict)
    images: list[Any] | None = None
    temperature: float | None = None


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
class Adapter:
    kind = "base"

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._model = None

    def load(self):  # pragma: no cover - heavy, runtime specific
        raise NotImplementedError

    def decide(self, req: SystemOneRequest) -> dict[str, Any]:
        raise NotImplementedError

    def ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self.load()
        return self._model

    def unload(self):
        with self._lock:
            self._model = None
            try:
                import torch  # noqa: F401

                torch.cuda.empty_cache()
            except Exception:
                pass


class QevAdapter(Adapter):
    """Base Labs models: OneJev qev runtime + serve.patch (readout.json)."""

    kind = "qev"

    def load(self):
        from qev.engine import DecisionEngine

        return DecisionEngine(str(self.path), device=DEVICE)

    def decide(self, req: SystemOneRequest) -> dict[str, Any]:
        from qev.schema import SystemOneRequest as QevRequest

        engine = self.ensure()
        payload = {"state": req.state, "questions": {k: v.model_dump(exclude_none=True) for k, v in req.questions.items()}}
        resp, extra = engine.decide(QevRequest(**payload))
        out = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        out.setdefault("answers", {})
        return out


class PplxAdapter(Adapter):
    """Perplexity Decider: ships its own pplx_decider package under source/."""

    kind = "pplx"

    def load(self):
        mode = os.environ.get("PPLX_MODE", "separate")      # separate | isolated
        fast = os.environ.get("PPLX_FAST", "0") == "1"      # packed path / CUDA graphs
        precision = os.environ.get("DECIDER_PRECISION", "auto").lower()
        is_nvfp4 = precision == "nvfp4" or (
            precision == "auto" and (self.path / "nvfp4_manifest.json").exists()
        )
        if is_nvfp4:
            from nvfp4.decider import NVFP4Decider

            decider = NVFP4Decider.from_pretrained(
                str(self.path), device=DEVICE, mode=mode, fast=fast
            )
            report = decider.runtime_report()
            print(
                "[gateway] nvfp4 runtime: "
                + json.dumps(
                    {
                        "checkpoint": report["checkpoint"]["root"],
                        "backend": report["backend"]["backend"],
                        "flashinfer": report["backend"]["flashinfer"],
                        "modules": report["nvfp4_modules"],
                        "kept_bf16_linears": len(report["kept_bf16_linears"]),
                        "fast": report["fast"],
                        "prefill_v2": report.get("prefill_v2", "off"),
                        "mode": report["mode"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if os.environ.get("DECIDER_NVFP4_WARMUP", "1") != "0":
                t0 = time.perf_counter()
                decider.decide(
                    {
                        "state": "warmup request for kernel compilation " * 4,
                        "questions": {"warmup": {"type": "noul"}},
                    }
                )
                if mode == "isolated":
                    decider.decide(
                        {
                            "state": "warmup request for isolated kernel compilation " * 4,
                            "questions": {
                                "warmup_1": {"type": "noul"},
                                "warmup_2": {"type": "noul"},
                            },
                        }
                    )
                print(f"[gateway] nvfp4 warmup {time.perf_counter() - t0:.1f}s", flush=True)
            return decider

        from pplx_decider import Decider

        return Decider.from_pretrained(str(self.path), device=DEVICE, mode=mode, fast=fast)

    def decide(self, req: SystemOneRequest) -> dict[str, Any]:
        decider = self.ensure()
        payload: dict[str, Any] = {
            "state": req.state,
            "questions": {k: v.model_dump(exclude_none=True) for k, v in req.questions.items()},
        }
        if req.images:
            payload["images"] = req.images
        cap = int(os.environ.get("PPLX_MAX_TOKENS", "0"))
        state = payload.get("state")
        if cap > 0 and isinstance(state, str) and len(state) > cap * 4:
            # keep the head and the tail: routing decisions rarely need the middle of a 20k document
            budget = cap * 4
            head, tail = int(budget * 0.7), budget - int(budget * 0.7)
            payload["state"] = state[:head] + "\n…\n" + state[-tail:]
        return decider.decide(payload)


class GlideAdapter(Adapter):
    """Fastino GLiDE: encoder + 255-way readout, ships the glide package."""

    kind = "glide"

    def load(self):
        from glide import Glide

        return Glide.from_pretrained(str(self.path))

    def decide(self, req: SystemOneRequest) -> dict[str, Any]:
        model = self.ensure()
        if req.images:
            raise HTTPException(400, "GLiDE has no vision encoder")
        return model.decide(
            state=req.state,
            questions={k: v.model_dump(exclude_none=True) for k, v in req.questions.items()},
        )


def detect(path: Path) -> Adapter:
    if (path / "glide").is_dir():
        return GlideAdapter(path)
    if (path / "source" / "src" / "pplx_decider").is_dir():
        return PplxAdapter(path)
    if (path / "readout.json").exists():
        return QevAdapter(path)
    raise ValueError(f"cannot detect decision-model runtime for {path}")


# --------------------------------------------------------------------------- #
# registry with simple LRU
# --------------------------------------------------------------------------- #
class Registry:
    def __init__(self, root: Path, only: str | None = None):
        self.root = root
        self.only = only
        self._lock = threading.Lock()
        self._adapters: dict[str, Adapter] = {}
        self._loaded: list[str] = []

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        names = sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "config.json").exists())
        if self.only:
            names = [n for n in names if n == self.only]
        return names

    def get(self, name: str | None) -> tuple[str, Adapter]:
        names = self.names()
        if not names:
            raise HTTPException(503, f"no models under {self.root} (only={self.only})")
        if self.only:
            # single-model container: accept any alias the gateway sends (new-api model rename etc.)
            short = self.only
        else:
            if name in (None, "", "auto", "default"):
                name = names[0]
            # allow "elephant-1" or "baselabs/elephant-1"
            short = name.split("/")[-1]
            if short not in names:
                raise HTTPException(404, f"unknown model {name!r}; available: {names}")
        with self._lock:
            if short not in self._adapters:
                self._adapters[short] = detect(self.root / short)
            adapter = self._adapters[short]
        return short, adapter

    def touch(self, name: str, adapter: Adapter):
        with self._lock:
            if name in self._loaded:
                self._loaded.remove(name)
            self._loaded.append(name)
            while MAX_LOADED > 0 and len(self._loaded) > MAX_LOADED:
                victim = self._loaded.pop(0)
                if victim != name:
                    self._adapters[victim].unload()


if os.environ.get("DEBUG_STACKS") == "1":
    import faulthandler

    faulthandler.dump_traceback_later(int(os.environ.get("DEBUG_STACKS_SEC", "45")), repeat=True)

REGISTRY = Registry(MODEL_ROOT, only=ONLY_MODEL)
app = FastAPI(title="decision-model gateway", version="1.0")


@app.on_event("startup")
def _preload():
    for name in PRELOAD:
        try:
            _, adapter = REGISTRY.get(name)
            adapter.ensure()
            REGISTRY.touch(name.split("/")[-1], adapter)
        except Exception as exc:  # pragma: no cover
            print(f"[gateway] preload {name} failed: {exc}", flush=True)


@app.get("/health")
def health():
    return {"ok": True, "models": REGISTRY.names(), "loaded": REGISTRY._loaded}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": n, "object": "model", "owned_by": "local", "kind": getattr(REGISTRY._adapters.get(n), "kind", None)}
            for n in REGISTRY.names()
        ],
    }


@app.post("/v1/systemone")
@app.post("/v1/decisions")
def systemone(req: SystemOneRequest):
    name, adapter = REGISTRY.get(req.model)
    t0 = time.perf_counter()
    try:
        result = adapter.decide(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    REGISTRY.touch(name, adapter)
    result.setdefault("id", f"decision-{uuid.uuid4().hex[:24]}")
    result.setdefault("object", "structured_decision")
    result.setdefault("created", int(time.time()))
    result["model"] = name
    result.setdefault("usage", {})
    result["usage"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse(result)


# --------------------------------------------------------------------------- #
# OpenAI-compatible facade, so an existing gateway (new-api / one-api /
# LiteLLM) can add these models as ordinary "OpenAI" channels.
# --------------------------------------------------------------------------- #
DEFAULT_QUESTIONS: dict[str, dict[str, Any]] = {
    "tier": {
        "type": "choice",
        "instructions": "Which model tier should answer this request?",
        "criteria": {
            "small": "simple lookup, chat, formatting, translation",
            "mid": "multi-step reasoning, code, tool use",
            "frontier": "hard math, long-context analysis, ambiguous or high-stakes",
        },
    },
    "escalate": {"type": "noul", "instructions": "Does this need a frontier model or a human?"},
}


class ChatMessage(BaseModel):
    role: str
    content: Any = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    # non-standard passthrough (new-api forwards unknown fields untouched)
    state: Any = None
    questions: dict[str, Question] | None = None
    stream: bool = False


def _extract(body: ChatRequest) -> tuple[Any, dict[str, Any]]:
    """state/questions from the extra fields, else from a JSON user message, else the transcript."""
    if body.state is not None and body.questions:
        return body.state, {k: v.model_dump(exclude_none=True) for k, v in body.questions.items()}

    for msg in reversed(body.messages):
        if msg.role != "user":
            continue
        text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict) and "questions" in parsed:
            questions = body.questions or parsed["questions"]
            if isinstance(questions, dict):
                questions = {k: (v if isinstance(v, dict) else {"type": str(v)}) for k, v in questions.items()}
            return parsed.get("state", parsed.get("input", text)), questions

    transcript = "\n".join(
        f"{m.role}: {m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)}"
        for m in body.messages
    )
    return transcript, {k: dict(v) for k, v in DEFAULT_QUESTIONS.items()}


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest):
    state, questions = _extract(body)
    req = SystemOneRequest(model=body.model, state=state,
                           questions={k: Question(**v) for k, v in questions.items()})
    name, adapter = REGISTRY.get(body.model)
    t0 = time.perf_counter()
    try:
        result = adapter.decide(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    REGISTRY.touch(name, adapter)

    answers = result.get("answers", result)
    content = json.dumps(answers, ensure_ascii=False)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    payload = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {**result.get("usage", {}),
                  "latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
    }
    if not body.stream:
        return JSONResponse(payload)

    def sse():
        first = {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                         "finish_reason": None}],
        }
        last = {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    from fastapi.responses import StreamingResponse

    return StreamingResponse(sse(), media_type="text/event-stream")


# expose qev's native schema when installed, so clients can use either shape
try:  # pragma: no cover
    from qev.schema import SystemOneRequest as QevRequest  # noqa: F401

    @app.post("/v1/systemone/qev")
    def systemone_qev(req: QevRequest):  # type: ignore[valid-type]
        name, adapter = REGISTRY.get(getattr(req, "model", None))
        return adapter.decide(SystemOneRequest(**req.model_dump()))
except Exception:
    pass
