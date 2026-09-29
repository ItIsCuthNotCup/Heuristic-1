"""heuristic-1: bonsai (thinker) + decider (judge) as one OpenAI-compatible model.

    thinker: bonsai 27B ternary, OpenAI-compatible llama-server .... :8010
    judge:   decider-4b, POST /v1/systemone ....................... :8008
    this module serves the merge as a single model on ............. :8200

The two halves stay in separate processes on purpose: bonsai needs the PrismML
llama.cpp GGUF path, decider needs a CUDA-graph engine, and they have
conflicting memory appetites on one GPU. The merge is an orchestration layer,
not a fused set of weights — one model to the caller, two processes underneath.

Licensing: this package is MIT. The two models it merges are Apache-2.0 and
are not redistributed here — Bonsai 2 27B (Prism ML) as the thinker, decider-4b
(Mapika) as the judge. The loop is MetaCog (MIT). See the repository NOTICE.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from metacog import Config, MetaCog, OpenAICompatThinker, SystemOneJudge

MODEL_ID = "heuristic-1"

__all__ = ["MODEL_ID", "build_mc", "clean_answer", "serve", "main"]


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def clean_answer(text: str) -> str:
    """Strip <think>...</think> reasoning blocks the thinker may leak."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def build_mc(**overrides: Any) -> MetaCog:
    """Assemble the merge.

    Defaults mirror the configuration the recorded HumanEval results were
    produced with: best-of-N over 8 paths, a greedy anchor in the pool, and a
    cascade that skips sampling when the greedy path already scores >= 0.95.

    Environment overrides: THINKER_URL, THINKER_MODEL, JUDGE_URL, N_PATHS,
    NO_CASCADE, MAX_TOKENS, TEMPERATURE.
    """
    cfg = dict(
        mode="best_of_n",
        strategy="noul",
        n_paths=int(env("N_PATHS", "8")),
        greedy_anchor=True,
        cascade_confidence=None if env("NO_CASCADE", "") == "1" else 0.95,
        max_tokens=int(env("MAX_TOKENS", "2048")),
        temperature=float(env("TEMPERATURE", "0.8")),
    )
    cfg.update(overrides)
    thinker = OpenAICompatThinker(
        env("THINKER_URL", "http://localhost:8010"),
        model=env("THINKER_MODEL", "bonsai"),
    )
    judge = SystemOneJudge(base_url=env("JUDGE_URL", "http://localhost:8008"))
    return MetaCog(thinker, judge, Config(**cfg))


def problem_from(messages: list[dict]) -> str:
    """Use the last user turn as the problem.

    MetaCog takes a bare problem string, so a multi-turn chat is folded down to
    its final user turn — the same convention the recorded runs used.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else json.dumps(c)
    return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)


def serve(mc: MetaCog, port: int) -> None:
    """Run the OpenAI-compatible server until interrupted."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "heuristic1/1.0"

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{time.strftime('%H:%M:%S')}] {self.command} {self.path}", flush=True)

        def do_GET(self) -> None:
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health", "/v1/models"):
                self._send(200, {"status": "ok", "model": MODEL_ID})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.rstrip("/")
            if path not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": "not found"})
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
                body = json.loads(raw)
            except Exception as e:  # noqa: BLE001 - report as a 400, never crash
                return self._send(400, {"error": f"bad JSON: {e}"})
            messages = body.get("messages", [])
            if not messages:
                return self._send(400, {"error": "messages required"})

            t0 = time.time()
            try:
                result = mc.run(problem_from(messages))
            except Exception as e:  # noqa: BLE001
                return self._send(502, {"error": f"pipeline failed: {e}"})
            dt = time.time() - t0

            rounds = result.trace.rounds
            conf = float(rounds[-1].verdict.confidence) if rounds else 0.0
            print(f"[{time.strftime('%H:%M:%S')}] conf={conf:.2f} {dt:.1f}s", flush=True)
            self._send(
                200,
                {
                    "id": f"merge-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", MODEL_ID),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": clean_answer(result.answer),
                            },
                            "finish_reason": "stop" if result.finished else "length",
                        }
                    ],
                    # completion_tokens counts the THINKER only. The judge's
                    # forward passes are not tokenised and are not included.
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": result.trace.thinker_tokens,
                        "total_tokens": result.trace.thinker_tokens,
                    },
                    "merge": {
                        "thinker": "bonsai",
                        "judge": "decider-4b",
                        "judge_confidence": round(conf, 4),
                        "seconds": round(dt, 2),
                        "thinker_calls": result.trace.thinker_calls,
                        "judge_calls": result.trace.judge_calls,
                        "rounds": len(rounds),
                    },
                },
            )

    print(f"{MODEL_ID} on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main() -> None:
    serve(build_mc(), int(env("PORT", "8200")))


if __name__ == "__main__":
    main()
