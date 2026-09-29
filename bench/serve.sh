#!/usr/bin/env bash
# Start heuristic-1 against the two servers it merges.
#
#   thinker  bonsai 27B ternary, llama-server, OpenAI-compatible .... :8010
#   judge    decider-4b, POST /v1/systemone ....................... :8008
#   merge    this repo ............................................ :8200
#
# Both halves must already be listening; this script only starts the merge and
# checks that the halves answer. Override with THINKER_URL / JUDGE_URL / PORT.
set -euo pipefail

THINKER_URL="${THINKER_URL:-http://localhost:8010}"
JUDGE_URL="${JUDGE_URL:-http://localhost:8008}"
PORT="${PORT:-8200}"

echo "thinker $THINKER_URL"
curl -sf "$THINKER_URL/props" >/dev/null && echo "  ok" || { echo "  NOT UP — start bonsai first"; exit 1; }
echo "judge   $JUDGE_URL"
curl -sf "$JUDGE_URL/health" >/dev/null && echo "  ok" || { echo "  NOT UP — start decider first"; exit 1; }

echo "starting heuristic-1 on :$PORT"
exec python -m heuristic1.server
