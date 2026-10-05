#!/usr/bin/env bash
# Download the Laya GGUF from Hugging Face. No account or token needed (public, Apache-2.0).
# Run with: bash scripts/get-model.sh [Q8_0|BF16]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUANT="${1:-Q8_0}"
FILE="Laya-$QUANT.gguf"
URL="https://huggingface.co/ggml-org/Laya-GGUF/resolve/main/$FILE"

mkdir -p "$ROOT/models"
cd "$ROOT/models"
if [[ -f "$FILE" ]]; then
  echo "$FILE already present"
else
  curl -L --fail --progress-bar -o "$FILE.part" "$URL"
  mv "$FILE.part" "$FILE"
fi
ls -lh "$FILE"
