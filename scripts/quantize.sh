#!/usr/bin/env bash
# Make a smaller Laya file from the full-precision one. Run with: bash scripts/quantize.sh [Q4_0|Q4_K_M]
# Q4_0 was the best trade-off in Phase 0 (252 MB, fastest, same accuracy as Q8_0 on the sample set).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUANTIZE="${QUANTIZE:-$HOME/build/llama.cpp/build/bin/llama-quantize}"
TYPE="${1:-Q4_0}"

[[ -f "$ROOT/models/Laya-BF16.gguf" ]] || bash "$ROOT/scripts/get-model.sh" BF16
"$QUANTIZE" "$ROOT/models/Laya-BF16.gguf" "$ROOT/models/Laya-$TYPE.gguf" "$TYPE"
sha256sum "$ROOT/models/Laya-$TYPE.gguf"
