#!/usr/bin/env bash
# Start a local llama-server with Laya for development. Run with: bash scripts/dev-server.sh
# (E: is mounted without exec permission, so call it through bash.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/build/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-4}"

if [[ -z "${MODEL:-}" ]]; then
  MODEL="$ROOT/models/Laya-Q4_0.gguf"
  [[ -f "$MODEL" ]] || MODEL="$ROOT/models/Laya-Q8_0.gguf"
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Download it with: bash scripts/get-model.sh" >&2
  exit 1
fi
echo "model: $MODEL" >&2

# Laya-Emoji is a classifier: llama-server returns its 1,977 scores from /embedding.
EXTRA=()
if [[ "$(basename "$MODEL" | tr 'A-Z' 'a-z')" == *emoji* ]]; then
  EXTRA=(--embedding --pooling rank)
fi

# Laya evaluates the whole prompt (question + options + message) in one batch of at most
# ~300 tokens, so 512 is plenty; one slot and no prompt cache keep memory low.
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -t "$THREADS" \
  -c 512 -b 512 -ub 512 -np 1 \
  --cache-ram 0 \
  --no-webui "${EXTRA[@]}"
