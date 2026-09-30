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

import hashlib
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from metacog import Config, MetaCog, OpenAICompatThinker, SystemOneJudge
from metacog.judge import TRIAGE_INSTRUCTIONS

MODEL_ID = "heuristic-1"

ANSWER_FIRST_SYSTEM = (
    "Answer first, explain after: lead with the final answer or complete code, "
    "then add any reasoning or notes below it. Never let explanation push the "
    "answer past the token limit."
)

__all__ = [
    "MODEL_ID",
    "ANSWER_FIRST_SYSTEM",
    "RESCUE_SUFFIX",
    "REPAIR_SUFFIX",
    "build_mc",
    "clean_answer",
    "code_block",
    "effort_for",
    "gate_log",
    "repair_answer",
    "rescue_answer",
    "serve",
    "main",
]


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def clean_answer(text: str) -> str:
    """Strip <think>...</think> reasoning blocks the thinker may leak."""
    out = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    # An unclosed trailing think block means the stream was cut mid-reasoning:
    # everything after the tag is think text, not an answer.
    return re.sub(r"<think>.*", "", out, flags=re.S).strip()


RESCUE_SUFFIX = (
    "\n\nYour analysis so far:\n{tail}\n\n"
    "Now output only the final answer or complete code in a single code "
    "block — no explanation."
)


def rescue_answer(thinker: Any, problem: str, winner_text: str) -> tuple[str, int]:
    """One low-effort continuation when the merge's winner is pure reasoning.

    Feeds the tail of the winning path's reasoning back and asks for the
    answer alone. Returns (answer text, tokens used); ("", 0) on failure.
    """
    tail = re.sub(r"</?think>", "", winner_text)[-1500:]
    try:
        gens = thinker.generate(
            problem + RESCUE_SUFFIX.format(tail=tail),
            "",
            n=1,
            max_tokens=int(env("RESCUE_MAX_TOKENS", "1024")),
            temperature=0.2,
        )
    except Exception as e:  # noqa: BLE001 - rescue is best-effort
        print(f"rescue failed: {e}", flush=True)
        return "", 0
    if not gens:
        return "", 0
    return gens[0].text, gens[0].tokens


_CODE_FENCE_RE = re.compile(r"```(\w*)\s*\n(.*?)```", re.S)


def code_block(text: str) -> tuple[str, str] | None:
    """First fenced code block as (language, body); None when absent."""
    m = _CODE_FENCE_RE.search(text)
    return (m.group(1).lower(), m.group(2)) if m else None


REPAIR_SUFFIX = (
    "\n\nThis code has a syntax error:\n```python\n{code}\n```\n"
    "Error: {error}\n\n"
    "Output only the corrected code in a single code block — no explanation."
)


def repair_answer(thinker: Any, problem: str, answer_text: str) -> tuple[str, int]:
    """One low-effort fix when the winner's fenced Python does not parse.

    Returns (repaired answer text, tokens used); ("", 0) when there is no
    broken Python block, the repair call fails, or the fix still does not
    parse. Only fires on answers that already contain a code block — an
    empty answer is the rescue path's job, not this one's.
    """
    block = code_block(answer_text)
    if block is None or block[0] not in ("", "python", "py"):
        return "", 0
    try:
        compile(block[1], "<answer>", "exec")
        return "", 0
    except SyntaxError as e:
        error = str(e)
    try:
        gens = thinker.generate(
            problem + REPAIR_SUFFIX.format(code=block[1][-1500:], error=error),
            "",
            n=1,
            max_tokens=int(env("REPAIR_MAX_TOKENS", "1024")),
            temperature=0.2,
        )
    except Exception as e:  # noqa: BLE001 - repair is best-effort
        print(f"repair failed: {e}", flush=True)
        return "", 0
    if not gens:
        return "", 0
    fixed = clean_answer(gens[0].text)
    new_block = code_block(fixed)
    if new_block is None:
        return "", gens[0].tokens
    try:
        compile(new_block[1], "<repaired>", "exec")
    except SyntaxError:
        return "", gens[0].tokens
    return fixed, gens[0].tokens


def build_mc(
    *,
    effort: str | None = None,
    answer_first: bool = False,
    **overrides: Any,
) -> MetaCog:
    """Assemble the merge.

    Defaults: best-of-N over 8 paths, a greedy anchor in the pool, and a cascade
    that skips sampling when the greedy path already scores >= CASCADE_CONFIDENCE
    (default 0.8; the recorded HumanEval rows used 0.95).

    ``effort`` sets the thinker's ``reasoning_effort`` chat-template field
    (xhigh/medium/low on bonsai); ``answer_first`` prepends a system prompt that
    puts the final answer or complete code before any explanation so a
    token-truncated reply still yields an answer.

    Environment overrides: THINKER_URL, THINKER_MODEL, JUDGE_URL, MODE, N_PATHS,
    NO_CASCADE, CASCADE_CONFIDENCE, MAX_TOKENS, TEMPERATURE, RACE_CONFIDENCE,
    RACE_SCORE_CHARS, SYSTEM_PROMPT, BRANCH_RACE, BRANCH_MIN_PATHS,
    BRANCH_MAX_TOKENS.
    """
    cfg = dict(
        mode=env("MODE", "best_of_n"),
        strategy="noul",
        n_paths=int(env("N_PATHS", "8")),
        greedy_anchor=True,
        cascade_confidence=None
        if env("NO_CASCADE", "") == "1"
        else float(env("CASCADE_CONFIDENCE", "0.8")),
        max_tokens=int(env("MAX_TOKENS", "2048")),
        temperature=float(env("TEMPERATURE", "0.8")),
        race_confidence=float(env("RACE_CONFIDENCE", "0.8")),
        race_score_chars=int(env("RACE_SCORE_CHARS", "600")),
        # branch-path upgrades (MetaCog v0.4): only paid when the cascade fails
        race_on_branch=env("BRANCH_RACE", "") == "on",
        branch_min_paths=int(v) if (v := env("BRANCH_MIN_PATHS", "")) else None,
        branch_max_tokens=int(v) if (v := env("BRANCH_MAX_TOKENS", "")) else None,
    )
    cfg.update(overrides)
    thinker = OpenAICompatThinker(
        env("THINKER_URL", "http://localhost:8010"),
        model=env("THINKER_MODEL", "bonsai"),
        extra_body={"reasoning_effort": effort} if effort else None,
        system_prompt=(env("SYSTEM_PROMPT", "") or (ANSWER_FIRST_SYSTEM if answer_first else ""))
        or None,
    )
    judge = SystemOneJudge(base_url=env("JUDGE_URL", "http://localhost:8008"))
    return MetaCog(thinker, judge, Config(**cfg))


def effort_for(judge: SystemOneJudge, problem: str) -> tuple[str, float]:
    """Pre-generation difficulty triage: judge the problem with an empty path
    and map P(easy) onto a chat-template ``reasoning_effort``.

    Effort is only *lowered* from the xhigh default when the judge is confident
    the problem is easy — a blind medium-effort pass measured worse on hard
    problems, so ambiguous scores stay at xhigh.
    """
    tv = judge.score(problem, [""], instructions=TRIAGE_INSTRUCTIONS)
    p_easy = (tv.raw or tv.probabilities)[0]
    low_min = float(env("EFFORT_LOW_MIN", "0.9"))
    med_min = float(env("EFFORT_MED_MIN", "0.7"))
    effort = "low" if p_easy >= low_min else "medium" if p_easy >= med_min else "xhigh"
    return effort, p_easy


_LOG_LOCK = threading.Lock()


def gate_log(path: str, row: dict) -> None:
    """Append one JSONL telemetry row (gate score + per-request stats).

    Joined with task outcomes on the bench side, these rows let us fit the
    cascade threshold that maximises accuracy per token instead of the
    guessed 0.8. Never raises: telemetry must not break serving."""
    try:
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"gate_log write failed: {e}", flush=True)


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


def serve(
    mc: MetaCog,
    port: int,
    *,
    effort_routing: bool | None = None,
    answer_first: bool | None = None,
    answer_rescue: bool | None = None,
    self_repair: bool | None = None,
    gate_log_path: str | None = None,
) -> None:
    """Run the OpenAI-compatible server until interrupted.

    ``effort_routing`` (env EFFORT_ROUTING=on) adds a cheap pre-generation
    judge triage and rebuilds the merge per request with the routed
    ``reasoning_effort``. ``answer_first`` (env ANSWER_FIRST=on) serves the
    answer-first system prompt. ``answer_rescue`` (env ANSWER_RESCUE=on)
    fires a single low-effort continuation when the winning path is pure
    reasoning with no answer. ``self_repair`` (env SELF_REPAIR=on) feeds a
    syntax-broken fenced code block back for one fix. ``gate_log_path``
    (env GATE_LOG) appends a JSONL telemetry row per request.
    """
    if effort_routing is None:
        effort_routing = env("EFFORT_ROUTING", "") == "on"
    if answer_first is None:
        answer_first = env("ANSWER_FIRST", "") == "on"
    if answer_rescue is None:
        answer_rescue = env("ANSWER_RESCUE", "") == "on"
    if self_repair is None:
        self_repair = env("SELF_REPAIR", "") == "on"
    if gate_log_path is None:
        gate_log_path = env("GATE_LOG", "") or None

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

            problem = problem_from(messages)
            effort: str | None = None
            triage: float | None = None
            use_mc = mc
            if effort_routing or answer_first:
                if effort_routing:
                    try:
                        effort, triage = effort_for(mc.judge, problem)
                    except Exception as e:  # noqa: BLE001 - triage is best-effort
                        print(f"triage failed, using xhigh: {e}", flush=True)
                        effort = "xhigh"
                use_mc = build_mc(effort=effort, answer_first=answer_first)

            t0 = time.time()
            try:
                result = use_mc.run(problem)
            except Exception as e:  # noqa: BLE001
                return self._send(502, {"error": f"pipeline failed: {e}"})
            answer_text = clean_answer(result.answer)
            rescued = False
            rescue_tokens = 0
            if answer_rescue and not answer_text:
                rescue_thinker = build_mc(effort="low").thinker
                rescued_text, rescue_tokens = rescue_answer(rescue_thinker, problem, result.answer)
                rescued_text = clean_answer(rescued_text)
                if rescued_text:
                    answer_text = rescued_text
                    rescued = True
            repaired = False
            repair_tokens = 0
            if self_repair and answer_text:
                repair_thinker = build_mc(effort="low").thinker
                repaired_text, repair_tokens = repair_answer(repair_thinker, problem, answer_text)
                if repaired_text:
                    answer_text = repaired_text
                    repaired = True
            dt = time.time() - t0

            rounds = result.trace.rounds
            conf = float(rounds[-1].verdict.confidence) if rounds else 0.0
            print(f"[{time.strftime('%H:%M:%S')}] conf={conf:.2f} {dt:.1f}s", flush=True)
            if gate_log_path:
                gate_log(
                    gate_log_path,
                    {
                        "ts": int(t0),
                        "problem_sha": hashlib.sha1(problem.encode()).hexdigest()[:12],
                        "conf": round(conf, 4),
                        "finished": result.finished,
                        "branched": result.trace.thinker_calls > 1,
                        "seconds": round(dt, 2),
                        "thinker_calls": result.trace.thinker_calls,
                        "judge_calls": result.trace.judge_calls,
                        "thinker_tokens": result.trace.thinker_tokens,
                        "rounds": len(rounds),
                        "mode": use_mc.config.mode,
                        "effort": effort,
                        "triage": triage,
                        "rescued": rescued,
                        "rescue_tokens": rescue_tokens,
                        "repaired": repaired,
                        "repair_tokens": repair_tokens,
                    },
                )
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
                                "content": answer_text,
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
                        "effort": effort,
                        "triage": triage,
                        "rescued": rescued,
                        "rescue_tokens": rescue_tokens,
                        "repaired": repaired,
                        "repair_tokens": repair_tokens,
                    },
                },
            )

    print(f"{MODEL_ID} on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main() -> None:
    answer_first = env("ANSWER_FIRST", "") == "on"
    serve(
        build_mc(answer_first=answer_first),
        int(env("PORT", "8200")),
        effort_routing=env("EFFORT_ROUTING", "") == "on",
        answer_first=answer_first,
        answer_rescue=env("ANSWER_RESCUE", "") == "on",
        self_repair=env("SELF_REPAIR", "") == "on",
        gate_log_path=env("GATE_LOG", "") or None,
    )


if __name__ == "__main__":
    main()
