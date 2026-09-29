# heuristic-1

**Bonsai 27B as thinker, decider-4b as judge — served as one OpenAI-compatible model.**

heuristic-1 runs a small local judge model over the reasoning a 27B model
produces, and returns the judge's pick. To the caller it is a single model on a
single endpoint. Underneath it is two GPU processes and an orchestration loop.

On HumanEval it scores **80.5 % against bonsai alone's 68.3 %** on the same 164
tasks, fixing 22 and breaking 2 (exact McNemar p = 0.000036), at roughly **7.9×
the thinker tokens**.

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
≥ 0.95 the samples are skipped, which keeps easy questions cheap.

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

Environment: `THINKER_URL`, `THINKER_MODEL`, `JUDGE_URL`, `PORT`, and the
generation knobs `N_PATHS`, `NO_CASCADE`, `MAX_TOKENS`, `TEMPERATURE`.

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
- **The gain is bought with tokens.** 7.9× is the price of 8 paths. `N_PATHS`
  is the dial if that is too much.
- **Not strictly dominant.** 2 tasks got worse. Adding paths can still lose.
- **The scaffold is a reconstruction.** The original evaluation script was not
  preserved anywhere on this machine. The dataset, the recorded rows, the
  prompt wording and the output schema were recovered, and the harness
  reproduces the recorded numbers exactly — but it was written afterwards, so
  agreement is necessary, not sufficient, evidence that it is the same script.

## Licensing

This package is MIT. The models are Apache-2.0: **Bonsai 27B** (Prism ML) and
**decider** (Mapika). The loop is from MetaCog (MIT). See `NOTICE`.
