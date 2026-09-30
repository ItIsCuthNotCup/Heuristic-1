# heuristic-1

**Bonsai 27B as thinker, decider-4b as judge — served as one OpenAI-compatible model.**

heuristic-1 runs a small local judge model over the reasoning a 27B model
produces, and returns the judge's pick. To the caller it is a single model on a
single endpoint. Underneath it is two GPU processes and an orchestration loop.

On HumanEval it scores **80.5 % against bonsai alone's 68.3 %** on the same 164
tasks, fixing 22 and breaking 2 (exact McNemar p = 0.000036). Since v1.2 the
cost profile changed: confident questions now cost ~1 generation, not ~8 —
measured **~8× faster and ~8× fewer tokens** on easy queries (see the cost
table below).

## What it is

```
client ──▶ :8200  heuristic-1        this repo, one /v1/chat/completions
              │
              ├──▶ :8010  bonsai 27B    thinker  (llama-server, OpenAI-compatible)
              │           writes N reasoning paths
              └──▶ :8008  decider-4b   judge    (POST /v1/systemone)
                          scores each path, the best one wins
```

The merge is an **orchestration layer, not fused weights**. The two halves stay
in separate processes on purpose: bonsai needs the PrismML llama.cpp GGUF path,
decider needs a CUDA-graph engine, and they have conflicting memory appetites
on one GPU. You get one model to talk to; nothing is quantised or merged into a
single artefact.

The loop itself is [MetaCog](https://github.com/ItIsCuthNotCup/MetaCog)
(MIT), vendored as a dependency. Configuration is best-of-N over 8 paths with a
greedy anchor in the pool and a cascade: if the greedy path already scores
≥ 0.8 (configurable) the samples are skipped, which keeps easy questions cheap.

## Results

HumanEval, 164 tasks, paired — same items, same prompt, same endpoints. The
oracle is the task's own unit test, executed in a subprocess with a CPU,
address-space and filesize cap.

| | bonsai alone | heuristic-1 |
|---|---|---|
| passed | 112/164 = 68.3 % | **132/164 = 80.5 %** |
| answer completed | 110/164 | 135/164 |
| thinker tokens (mean) | 708 | 5 564 |
| thinker generations (mean) | 1.00 | 1.98 |
| judge calls (mean) | 0 | 8.53 |

**Fixed 22 / lost 2, exact McNemar p = 0.000036.**

Read the token column before celebrating: the +12.2 points costs 7.9× the
generation budget. `usage.completion_tokens` counts the thinker only — the
judge's ~8.5 forward passes are not tokenised and do not appear there.

`runs/he_bonsai_full.jsonl` and `runs/he_merge_full.jsonl` are the raw recorded
rows, committed so the numbers above can be checked rather than trusted.

## What v1.2 actually costs

Live measurements on the production setup (bonsai on a DGX Spark, decider on
its own engine, same question in every arm):

| | wall | thinker tokens |
|---|---:|---:|
| heuristic-1 v1.0 (cascade 0.95, best-of-8) | 72.9 s | 2 101 |
| **v1.2, `CASCADE_CONFIDENCE=0.8` (default)** | **9.0 s** | **251** |
| v1.2, `MODE=race` | 17.4 s | 251 |
| bonsai alone (no merge) | 8.9 s | 251 |

Easy question, judge confident ≥0.8 → the greedy answer ships immediately and
the merge costs ~1 generation, same as calling bonsai directly. That is the
common case: most of the time the judge's confidence gate makes the merge
nearly free.

What still costs:

- **Unsure questions still pay for paths.** When greedy scores <0.8 the merge
  samples up to `N_PATHS` alternatives — that is the accuracy-vs-cost dial,
  and it is where the +12 points on HumanEval comes from.
- **`MODE=race` needs server parallelism.** With bonsai at llama-server's
  default single slot the sampled paths queue serially and race can only kill
  what has not run yet — on hard questions that is as slow as v1.0. Launch
  bonsai with `--parallel N` (≥ `N_PATHS`) and race drops hard questions to
  ~one generation's wall-clock while losers are cancelled mid-flight.
- **Judge calls aren't in `usage`.** decider's forwards are real compute that
  never appears in the token counts.

## Install

```bash
pip install -e .
```

The `metacog` dependency is pinned to git. **Do not `pip install metacog`** —
that name on PyPI belongs to an unrelated package (Metacog AI SDK).

## Run

Start the two halves first — bonsai on :8010 and decider on :8008, per their own
upstream instructions. Then:

```bash
bench/serve.sh              # checks both halves are up, then serves :8200
```

Or directly:

```bash
python -m heuristic1.server
```

Environment: `THINKER_URL`, `THINKER_MODEL`, `JUDGE_URL`, `PORT`, `MODE`, and the
generation knobs `N_PATHS`, `NO_CASCADE`, `CASCADE_CONFIDENCE`, `MAX_TOKENS`,
`TEMPERATURE`, `RACE_CONFIDENCE`, `RACE_SCORE_CHARS`.

v1.3 knobs (all opt-in, off by default):

- `EFFORT_ROUTING=on` — a cheap pre-generation judge call scores how easy the
  problem looks and routes the thinker's `reasoning_effort` (low/medium/xhigh,
  bonsai's chat-template field). Effort is only *lowered* from the xhigh
  default when the judge is confident the problem is easy — a blind
  medium-effort pass measured worse on hard problems. `EFFORT_LOW_MIN`
  (default 0.9) and `EFFORT_MED_MIN` (default 0.7) set the triage thresholds.
- `ANSWER_FIRST=on` — adds a system prompt putting the final answer / complete
  code before any explanation, so a MAX_TOKENS-truncated reply still yields an
  answer (two v1.2 tasks returned empty exactly this way). `SYSTEM_PROMPT`
  overrides the stock text.
- `ANSWER_RESCUE=on` — when the winning path is pure reasoning with no
  answer (the "all-think winner" failure mode), fires one low-effort
  continuation that feeds the reasoning tail back and asks for the answer
  alone (`RESCUE_MAX_TOKENS`, default 1024). Only fires on otherwise-empty
  answers, so it is nearly free.
- `GATE_LOG=/path.jsonl` — appends one telemetry row per request (gate score,
  branch decision, tokens, calls, effort, triage score, rescue use) so the
  cascade threshold can be fitted from data instead of guessed.

## Making it faster

The merge's cost is `N_PATHS` generations per question, so speed work is about
spending fewer tokens or running the paths at once.

- **`MODE=race`** — the biggest change. Every path streams concurrently and
  decider re-scores the partial text; the first stream to hit
  `RACE_CONFIDENCE` (default 0.8) wins and the losers are cancelled mid-flight.
  Wall-clock drops to ~one generation instead of two serial rounds, and losing
  paths stop paying tokens early. Two paths agreeing on an answer also end the
  race early. Requires bonsai's llama-server to run the streams in parallel —
  launch it with `--parallel N` (≥ `N_PATHS`) or the win collapses.
- **`CASCADE_CONFIDENCE=0.8`** (now the default) — the greedy path alone answers
  the question when the judge scores it ≥0.8; only unsure questions pay for
  extra paths. (The recorded HumanEval rows above used 0.95.)
- **`N_PATHS=4`** — halves sampling cost; the dial for accuracy vs speed.
- **`MAX_TOKENS=1000`** — caps each path shorter; most fixes fit.
- **decider on CPU** — it's a 4B; isolating it keeps its judge forwards off
  bonsai's GPU bandwidth.

Then talk to it like any OpenAI-compatible endpoint:

```bash
curl http://localhost:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"heuristic-1","messages":[{"role":"user","content":"..."}]}'
```

## Reproducing the numbers

```bash
python bench/run_humaneval.py --arms bonsai,merge --out runs
```

`HUMANEVAL_JSON` points at the 164-task dataset
(`task_id`, `prompt`, `test`, `entry_point`); it defaults to
`~/jev-chain/data/humaneval_test.json`.

**Grading policy, stated explicitly because it decides the numbers.** A reply
counts only if it contains a fenced code block — the prompt asks for the whole
function in one ` ```python ` block, so prose or a reasoning fragment that never
reached a fence is an undelivered answer. `finished` is recorded but not used to
gate the score. Trailing lines are trimmed until the candidate parses.

Regrading the committed rows under this policy reproduces both recorded
accuracies exactly: **112/164 and 132/164, 164/164 row-level agreement on each
arm.** The policy was recovered by fitting against the recorded results, not
guessed — the alternatives (stripping `<think>` first, or rejecting truncated
replies) each move the totals, and none reproduces the originals.

## Honest limits

- **HumanEval only.** These are code-generation tasks graded by unit tests.
  Nothing here supports a claim about math, GPQA, or general chat, and the
  `usage` numbers say nothing about the judge's own compute cost.
- **The gain is bought with tokens.** The recorded 7.9× is the old config
  (cascade 0.95, which almost never fired). With the v1.2 0.8 cascade, easy
  questions cost ~1 generation; unsure ones still pay for paths. `N_PATHS`
  is the dial.
- **Not strictly dominant.** 2 tasks got worse. Adding paths can still lose.
- **The scaffold is a reconstruction.** The original evaluation script was not
  preserved anywhere on this machine. The dataset, the recorded rows, the
  prompt wording and the output schema were recovered, and the harness
  reproduces the recorded numbers exactly — but it was written afterwards, so
  agreement is necessary, not sufficient, evidence that it is the same script.

## Credits

heuristic-1 is a thin layer over other people's work. None of the intelligence
here is ours.

- **[Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)** —
  Prism ML, Apache-2.0. The thinker. 27B reasoning model in ternary weights,
  ~1.72 bits/weight, 7.21 GB as PQ2_0 GGUF.
- **[decider](https://huggingface.co/Mapika/decider-4b)** — Mapika
  (Mark Marosi), Apache-2.0. The judge. A 4B model that reads a state and typed
  questions and returns calibrated probabilities in one forward pass.
- **[MetaCog](https://github.com/ItIsCuthNotCup/MetaCog)** — Jacob Cuthbertson,
  MIT. The metacognition loop: the best-of-N search, the greedy anchor and the
  cascade that make this a merged model rather than a single generation.
- **[llama.cpp](https://github.com/PrismML-Eng/llama.cpp)** (PrismML fork, MIT) —
  the runtime serving the thinker. Stock llama.cpp will not load these files.
- **[HumanEval](https://arxiv.org/abs/2107.03374)** — Chen et al., 2021. The
  benchmark and its unit-test oracle.

If you use this, cite Bonsai and decider — both upstream projects ask for it,
and the credits are theirs:

```bibtex
@techreport{bonsai2_27b,
  title  = {Bonsai 2 27B: A 27B Ternary Reasoning Model},
  author = {Prism ML}, year = {2026}, month = {September},
  url    = {https://prismml.com}
}
@software{marosi2026decider,
  author = {Marosi, Mark},
  title  = {decider: one-pass typed decisions with calibrated probabilities},
  year   = {2026}, url = {https://github.com/Mapika/decider}
}
```

Full attributions, including the Qwen base models both fine-tunes derive from,
are in [`NOTICE`](NOTICE).

## Licensing

This package is MIT — that covers the code in `heuristic1/` and `bench/`, and
nothing else. The models are Apache-2.0 and are **not redistributed here**;
operators obtain and serve them themselves. `NOTICE` records the attributions.
