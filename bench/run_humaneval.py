#!/usr/bin/env python3
"""HumanEval A/B: bonsai alone vs heuristic-1, paired on the same 164 tasks.

    python bench/run_humaneval.py --arms bonsai,merge --out runs

Two arms, identical items and identical prompt:

  bonsai   one greedy (temperature 0) generation from bonsai :8010
  merge    heuristic-1 :8200 — bonsai writes 8 paths, decider judges, best wins

Each answer is graded by executing the candidate against the task's own test in
a subprocess with a CPU/address-space/filesize cap, so the oracle is the unit
test, not a string match.

Reported per arm: pass rate, mean thinker tokens, mean generations and mean
judge calls. The head-to-head is fixed/lost on identical task ids with an exact
McNemar test — a bare accuracy delta on 164 items is not evidence.

    python bench/run_humaneval.py --out runs/he_check
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INSTRUCTION = "Complete the following Python function. Return the full function in a single ```python code block."
PASS_MARKER = "__PASS__"
DEFAULT_TIMEOUT = 10
MAX_AS = 2 * 1024**3  # 2 GiB address-space cap per candidate

DATA = Path(os.environ.get("HUMANEVAL_JSON", Path.home() / "jev-chain" / "data" / "humaneval_test.json"))


# ------------------------------------------------------------- extraction ---

def _from_fence(text: str) -> str | None:
    """Return the contents of the LAST ``` fenced block, fences stripped."""
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.S)
    if not blocks:
        return None
    return blocks[-1]


def _parse_clean(src: str) -> str | None:
    """Drop trailing lines until the source parses, up to 40 times."""
    lines = src.splitlines()
    for _ in range(40):
        if not lines:
            return None
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines)
        except SyntaxError as e:
            cut = (e.lineno or len(lines)) - 1
            lines = lines[:cut] if 0 < cut < len(lines) else lines[:-1]
    return None


def extract_code(text: str, item: dict) -> str | None:
    """Pull a runnable program out of a model reply.

    Handles both reply styles: a complete function in a fence, or a bare body
    that continues the signature we showed. Imports directly above the entry
    point are kept, since the prompt's own annotations need them.

    A reply with no fenced block scores nothing. The prompt asks for the whole
    function in one ```` ```python ```` block, so prose or a reasoning fragment
    that never reached a fence is an undelivered answer, not a body to splice
    onto the prompt. This is the policy that reproduces the recorded results
    exactly — see README, "Reproducing the numbers".
    """
    if not text:
        return None
    body = _from_fence(text)
    if body is None:
        return None
    ep = item["entry_point"]
    lines = body.splitlines()
    idx = next((i for i, l in enumerate(lines)
                if re.match(rf"\s*def\s+{re.escape(ep)}\s*\(", l)), None)
    if idx is not None:
        start = idx
        while start > 0 and (not lines[start - 1].strip()
                             or re.match(r"\s*(import|from)\s", lines[start - 1])):
            start -= 1
        return _parse_clean("\n".join(lines[start:]).strip("\n"))
    if lines and (lines[0].startswith((" ", "\t"))
                  or any(l.strip().startswith(("return", "def ", "if ", "for ", "while "))
                         for l in lines)):
        return _parse_clean(item["prompt"].rstrip() + "\n" + body)
    return _parse_clean(body)


# ----------------------------------------------------------------- grading ---

RUNNER = """\
import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_AS, ({maxas}, {maxas}))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
sys.stdout = open(1, "w", buffering=1)
src = sys.stdin.read()
ns = {{}}
exec(compile(src, "<cand>", "exec"), ns)
"""


def run_tests(program: str, item: dict, timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    """Execute the candidate against the task's test. Returns (passed, error)."""
    harness = item["test"].rstrip() + "\n\n"
    harness += f"check({item['entry_point']})\nprint({PASS_MARKER!r})\n"
    source = program.rstrip() + "\n\n" + harness
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cand.py")
        with open(path, "w") as fh:
            fh.write(RUNNER.format(cpu=timeout + 2, maxas=MAX_AS))
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", path],
                input=source, text=True, capture_output=True,
                timeout=timeout, cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: exceeded {timeout}s"
    if PASS_MARKER in proc.stdout:
        return True, ""
    err = [l for l in proc.stderr.strip().splitlines() if l.strip()]
    return False, (err[-1][:300] if err else f"rc={proc.returncode}")


# ------------------------------------------------------------------ client ---

def chat(url: str, model: str, prompt: str, *, temperature: float, max_tokens: int,
         api_key: str = "") -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "heuristic1-bench/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=int(os.environ.get("BENCH_TIMEOUT", "3600"))) as r:
        return json.load(r)


def build_prompt(item: dict) -> str:
    return f"{INSTRUCTION}\n\n{item['prompt']}"


def run_one(url: str, model: str, item: dict, arm: str, max_tokens: int) -> dict:
    t0 = time.time()
    try:
        resp = chat(url, model, build_prompt(item),
                    temperature=0.0 if arm in ("bonsai", "raw") else 0.8,
                    max_tokens=max_tokens,
                    api_key=os.environ.get("RAW_API_KEY", "") if arm == "raw" else "")
    except Exception as e:  # noqa: BLE001 - a dead row is data, not a crash
        return {"arm": arm, "task_id": item["task_id"], "passed": False,
                "answer": "", "finished": False,
                "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or ""
    if not isinstance(text, str):
        text = str(text)
    finished = (resp.get("choices") or [{}])[0].get("finish_reason") == "stop"

    program = extract_code(text, item)
    if program is None:
        return {"arm": arm, "task_id": item["task_id"], "passed": False,
                "answer": text[:4000], "finished": finished,
                "error": "no code extracted", "seconds": time.time() - t0}

    passed, err = run_tests(program, item)
    trace = {"rounds": 1, "thinker_calls": 1, "judge_calls": 0,
             "thinker_tokens": int((resp.get("usage") or {}).get("completion_tokens") or 0)}
    merge_meta = resp.get("merge")
    if merge_meta:
        trace = {
            "rounds": int(merge_meta.get("rounds", 1)),
            "thinker_calls": int(merge_meta.get("thinker_calls", 1)),
            "judge_calls": int(merge_meta.get("judge_calls", 0)),
            "thinker_tokens": trace["thinker_tokens"],
        }
    return {"arm": arm, "task_id": item["task_id"], "passed": passed,
            "answer": text[:20000], "finished": finished,
            "error": err, "seconds": time.time() - t0, "trace": trace}


# -------------------------------------------------------------- statistics ---

def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


# -------------------------------------------------------------------- main ---

ARMS = {
    "bonsai": ("http://localhost:8010/v1/chat/completions", "bonsai"),
    "merge": ("http://localhost:8200/v1/chat/completions", "heuristic-1"),
    # any hosted OpenAI-compatible model alone, e.g. to pair against a merge
    # server whose THINKER_URL points at the same model
    "raw": (os.environ.get("RAW_URL", ""), os.environ.get("RAW_MODEL", "")),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="bonsai,merge")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 164")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="output cap per request; use the same value for every arm")
    args = ap.parse_args()

    items = json.load(open(DATA))
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        url, model = ARMS[arm]
        path = out / f"he_{arm}_full.jsonl"
        t0 = time.time()
        print(f"[{arm}] {len(items)} tasks -> {path}", flush=True)
        with path.open("w") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, url, model, it, arm, args.max_tokens) for it in items]
            for i, f in enumerate(futs, 1):
                rec = f.result()
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(items):
                    print(f"  {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)

    report(out, items, [a.strip() for a in args.arms.split(",") if a.strip()])


def report(out: Path, items: list[dict], arms: list[str]) -> None:
    data = {}
    for arm in arms:
        p = out / f"he_{arm}_full.jsonl"
        if p.exists():
            data[arm] = {r["task_id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}

    print("\n" + "=" * 76)
    print(f"{'arm':<10}{'n':>5}{'passed':>9}{'acc':>9}{'95% CI':>17}{'tok/item':>10}{'sec/item':>10}")
    print("-" * 76)
    for arm in arms:
        rs = data.get(arm, {})
        if not rs:
            continue
        k = sum(1 for r in rs.values() if r.get("passed"))
        n = len(rs)
        lo, hi = wilson(k, n)
        toks = [r.get("trace", {}).get("thinker_tokens", 0) for r in rs.values()]
        secs = [r.get("seconds", 0.0) for r in rs.values()]
        print(f"{arm:<10}{n:>5}{k:>9}{100 * k / n:>8.1f}%"
              f"{f'[{100*lo:.1f}, {100*hi:.1f}]':>17}"
              f"{sum(toks) / n:>10.0f}{sum(secs) / n:>10.1f}")

    base = "raw" if "raw" in data else "bonsai"
    if base in data and "merge" in data:
        b, m = data[base], data["merge"]
        shared = sorted(set(b) & set(m))
        fixed = sum(1 for i in shared if m[i].get("passed") and not b[i].get("passed"))
        lost = sum(1 for i in shared if b[i].get("passed") and not m[i].get("passed"))
        p = mcnemar(fixed, lost)
        print("-" * 76)
        print(f"paired on {len(shared)} tasks · fixed {fixed} / lost {lost} · exact McNemar p = {p:.6f}")
        print(f"verdict: {'heuristic-1 is BETTER' if p < 0.05 and fixed > lost else 'no significant difference'}")
    print("=" * 76)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
