#!/bin/bash
# Checks the Nebius Token Factory key from .env.
#   ./tools/tokenfactory_check.sh          list available models (free, no tokens used)
#   ./tools/tokenfactory_check.sh ask "…"  one short chat question (costs a few tokens)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && set -a && source .env && set +a
[ -n "${NEBIUS_API_KEY:-}" ] || { echo "NEBIUS_API_KEY fehlt in .env" >&2; exit 1; }
BASE=https://api.tokenfactory.nebius.com/v1

if [ "${1:-}" = "ask" ]; then
    MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
    jq -n --arg m "$MODEL" --arg q "${2:-Sag kurz Hallo.}" \
        '{model: $m, max_tokens: 200, messages: [{role: "user", content: $q}]}' |
    curl -sS "$BASE/chat/completions" -H "Authorization: Bearer $NEBIUS_API_KEY" \
        -H 'Content-Type: application/json' --data-binary @- |
    jq -r '.choices[0].message.content // .'
else
    curl -sS "$BASE/models" -H "Authorization: Bearer $NEBIUS_API_KEY" | jq -r '.data[].id // .' | sort
fi
