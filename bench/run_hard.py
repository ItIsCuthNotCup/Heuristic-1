#!/usr/bin/env python3
"""Hard-set A/B: a hosted model alone vs heuristic-1 over the same model.

    RAW_URL=https://api.commandcode.ai/provider/v1/chat/completions \
    RAW_MODEL=stealth/space-bunny-alpha RAW_API_KEY=... \
    python bench/run_hard.py --arms raw,merge --max-tokens 16000 --out runs/hard

Sets: GPQA Diamond (60, multiple choice) and AIME 2024-2025 (60, integer).
Both arms get the identical user prompt and the identical per-request output
cap. ``raw`` is one temperature-0 completion; ``merge`` is the local merge
server (:8200) whose THINKER_URL points at the same model. Graded by the last
\\boxed{} in the reply against gold.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_humaneval import mcnemar, wilson

BENCH = Path(os.environ.get("BENCH_DIR", Path.home() / "repos" / "MetaCog" / "runs" / "bench"))
SETS = {
    "gpqa": BENCH / "gpqa_diamond_60.jsonl",
    "aime": BENCH / "aime_2024_2025.jsonl",
}
AIME_SUFFIX = "\n\nPlease write your final answer as an integer in the form \\boxed{N}."


def load(name: str) -> list[dict]:
    rows = [json.loads(line) for line in open(SETS[name]) if line.strip()]
    for r in rows:
        r["set"] = name
        if name == "aime":
            r["problem"] += AIME_SUFFIX
    return rows


def boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return m[-1].strip() if m else None


def correct(item: dict, got: str | None) -> bool:
    if got is None:
        return False
    if item["set"] == "gpqa":
        return got.strip("() ").upper() == item["answer"]
    try:
        return int(float(got.replace(",", ""))) == int(item["answer"])
    except ValueError:
        return False


def chat(url: str, model: str, prompt: str, *, temperature: float, max_tokens: int,
         api_key: str) -> dict:
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


ARMS = {
    "raw": (os.environ.get("RAW_URL", ""), os.environ.get("RAW_MODEL", ""), 0.0),
    "merge": ("http://localhost:8200/v1/chat/completions", "heuristic-1", 0.0),
}


def run_one(arm: str, item: dict, max_tokens: int) -> dict:
    url, model, temp = ARMS[arm]
    key = os.environ.get("RAW_API_KEY", "") if arm == "raw" else ""
    t0 = time.time()
    rec = {"arm": arm, "set": item["set"], "id": item["id"], "want": item["answer"]}
    for attempt in range(3):
        try:
            resp = chat(url, model, item["problem"], temperature=temp,
                        max_tokens=max_tokens, api_key=key)
            if "error" in resp:
                raise RuntimeError(str(resp["error"])[:300])
            break
        except Exception as e:  # noqa: BLE001 - a dead row is data, not a crash
            if attempt == 2:
                return {**rec, "ok": False, "got": None, "tokens": 0,
                        "finished": False, "error": f"{type(e).__name__}: {e}",
                        "seconds": time.time() - t0}
            time.sleep(10 * (attempt + 1))
    choice = (resp.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    got = boxed(text)
    return {**rec, "ok": correct(item, got), "got": got,
            "tokens": int((resp.get("usage") or {}).get("completion_tokens") or 0),
            "finished": choice.get("finish_reason") == "stop",
            "seconds": round(time.time() - t0, 1),
            "merge": resp.get("merge"), "answer": text[-3000:]}


def report(out: Path, arms: list[str]) -> None:
    data = {}
    for arm in arms:
        p = out / f"{arm}.jsonl"
        if p.exists():
            data[arm] = {(r["set"], r["id"]): r for r in map(json.loads, open(p))}
    for s in ("gpqa", "aime", "all"):
        print(f"\n== {s} ==")
        print(f"{'arm':<8}{'n':>5}{'pass':>6}{'acc':>8}{'95% CI':>16}"
              f"{'tok/q':>9}{'sec/q':>8}{'capped':>8}")
        sub = {a: {k: r for k, r in d.items() if s == "all" or k[0] == s} for a, d in data.items()}
        for arm, rs in sub.items():
            n = len(rs)
            if not n:
                continue
            k = sum(r["ok"] for r in rs.values())
            lo, hi = wilson(k, n)
            print(f"{arm:<8}{n:>5}{k:>6}{100*k/n:>7.1f}%{f'[{100*lo:.0f}, {100*hi:.0f}]':>16}"
                  f"{sum(r['tokens'] for r in rs.values())/n:>9.0f}"
                  f"{sum(r['seconds'] for r in rs.values())/n:>8.1f}"
                  f"{sum(not r['finished'] for r in rs.values()):>8}")
        if "raw" in sub and "merge" in sub:
            shared = sorted(set(sub["raw"]) & set(sub["merge"]))
            fixed = sum(sub["merge"][i]["ok"] and not sub["raw"][i]["ok"] for i in shared)
            lost = sum(sub["raw"][i]["ok"] and not sub["merge"][i]["ok"] for i in shared)
            print(f"paired n={len(shared)}  fixed {fixed} / lost {lost}  "
                  f"McNemar p={mcnemar(fixed, lost):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="raw,merge")
    ap.add_argument("--sets", default="gpqa,aime")
    ap.add_argument("--limit", type=int, default=0, help="per set; 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="output cap per request; the same for every arm")
    ap.add_argument("--out", default="runs/hard")
    ap.add_argument("--report", action="store_true", help="only print the report")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    if not args.report:
        items = []
        for s in args.sets.split(","):
            rows = load(s)
            items += rows[: args.limit] if args.limit else rows
        for arm in arms:
            path = out / f"{arm}.jsonl"
            done = ({(r["set"], r["id"]) for r in map(json.loads, open(path))}
                    if path.exists() else set())
            todo = [it for it in items if (it["set"], it["id"]) not in done]
            print(f"[{arm}] {len(todo)} to run -> {path}", flush=True)
            with path.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
                for rec in pool.map(lambda it, a=arm: run_one(a, it, args.max_tokens), todo):
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"  {rec['set']} {rec['id']} ok={rec['ok']} got={rec['got']} "
                          f"tok={rec['tokens']} {rec['seconds']}s", flush=True)
    report(out, arms)


if __name__ == "__main__":
    main()
