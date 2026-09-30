#!/usr/bin/env python3
"""Subset eval: run a fixed task list against any merge endpoint.

    python bench/subset_eval.py --url http://localhost:8200 \
        --tasks HumanEval/9 HumanEval/29 ... --out runs/subset_v13.jsonl

Imports run_one from run_humaneval so extraction and grading are identical
to the recorded runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_humaneval import DATA, run_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="heuristic-1")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = {it["task_id"]: it for it in json.load(open(DATA))}
    missing = [t for t in args.tasks if t not in items]
    if missing:
        print("unknown task ids:", missing)
        sys.exit(1)

    url = args.url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    with open(args.out, "w") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, url, args.model, items[t], "merge"): t for t in args.tasks}
        done = 0
        for f in futs:
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            mark = "P" if rec.get("passed") else "F"
            tok = rec.get("trace", {}).get("thinker_tokens", 0)
            print(
                f"  {done}/{len(args.tasks)} {mark} {rec['task_id']} "
                f"{tok}tok {rec.get('seconds', 0):.0f}s",
                flush=True,
            )
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
