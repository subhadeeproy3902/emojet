#!/usr/bin/env bash
# Build the Laya engine (llama-server) at the pinned llama.cpp commit with EmoJet's patches.
# Run with: bash scripts/build-engine.sh [host|android|all]   (default: all)
#   host    -> $LLAMA/build/bin/llama-server            (laptop: dev-server.sh, tests)
#   android -> $LLAMA/build-android/bin/llama-server    (arm64 phones, KleidiAI; packaged by build-android.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAT="${1:-all}"
LLAMA="${LLAMA:-$HOME/build/llama.cpp}"
COMMIT="2e7c58c5477478c8cf6e199cfaa5dcd5a4319c81"   # first commit with Laya (/v1/systemone), tested 2026-10-05
NDK="${NDK:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/30.0.16248370}"

if [[ ! -d "$LLAMA/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
if [[ "$(git -C "$LLAMA" rev-parse HEAD)" != "$COMMIT" ]]; then
  git -C "$LLAMA" fetch --depth 1 origin "$COMMIT"
  git -C "$LLAMA" checkout -q "$COMMIT"
fi

# Apply each patch once (skip if it's already in the tree).
for p in "$ROOT"/engine/llama.cpp-patches/*.patch; do
  if git -C "$LLAMA" apply --reverse --check "$p" 2>/dev/null; then
    echo "patch already applied: $(basename "$p")"
  else
    git -C "$LLAMA" apply "$p" && echo "patch applied: $(basename "$p")"
  fi
done

if [[ "$WHAT" == host || "$WHAT" == all ]]; then
  [[ -f "$LLAMA/build/CMakeCache.txt" ]] || cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  nice -n 10 cmake --build "$LLAMA/build" -j "$(nproc)" --target llama-server llama-quantize
fi

if [[ "$WHAT" == android || "$WHAT" == all ]]; then
  [[ -f "$LLAMA/build-android/CMakeCache.txt" ]] || cmake -S "$LLAMA" -B "$LLAMA/build-android" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
    -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF
  nice -n 10 cmake --build "$LLAMA/build-android" -j "$(nproc)" --target llama-server
fi
