# Release — Emojet v1 (2026-10-24)

**Published.** The model is live at <https://huggingface.co/subhadeep3902/emojet>.

| | |
|---|---|
| Public name | **Emojet v1** |
| Weights | `model.safetensors` + `Emojet-v1-F16.gguf` + `Emojet-v1-Q4_K_M.gguf` (274 MB, the phone file) |
| Licence | weights CC BY-NC-SA 4.0 · code Apache-2.0 · base model Laya Apache-2.0 |
| Pinned revision used by the app | `d77342b5fe181f13894fd93639f37d1eecb23df3` |
| Q4_K_M sha256 | `b255d653b913e4f873f9af3ebb0403873e6c98850ed11605a3ca8c76015dee78` |
| App release | `emojet-ai` v1.0.0, APK 8 MB (arm64) |

## What v1 is, internally

v1 is the **weight average (α = 0.5)** of two models:

- `emojet-v2-round2` — the real-message model (2.9 M messages, EN/HI/BN).
- `emojet-v2-round3b` — the same model continued on the chat lexicon, meaning-aware labels and synthetic chat lines.

The blend was chosen by `training/build_soup.py`: every ratio from 0.5 to 1.0 was evaluated, and the one with the
most slang skill that had **no statistically significant loss** on real held-out messages (paired McNemar per
language) was kept. See `docs/TRAINING.md` for the whole history, including the rounds that regressed.

## Internal name map

| Internal | Kaggle kernel | Public |
|---|---|---|
| Emojet-v1 (46 tones, 4 styles) | `emojet-model-train`, `emojet-v1-resume` | not released |
| Emojet-v1-mini | `emojet-v1-mini*` | not released |
| Emojet-v2-pre | `emojet-v2` | not released |
| Emojet-v2 (real data) | `emojet-v2-real` | not released |
| Emojet-v2 round 2 | `emojet-v2-round2` | half of v1 |
| Emojet-v2 round 3 / 3b | `emojet-v2-round3`, `emojet-v2-round3b` | 3b is the other half of v1 |
| **blend α = 0.5** | `emojet-v2-soup` | **Emojet v1** |

Public version numbers start at 1 and count from here.

## Checklist (done)

- [x] Beats the previous model on slang (29 → 41/72 on the phone file) with no significant regression
- [x] Model card with metrics, limitations, licence and both quick starts
- [x] `labels.json` (2,002 outputs in order) and `emojet.json` (metadata + metrics)
- [x] GGUF Q4_K_M verified byte-identical to the local file after upload
- [x] App `shared/models.json` points at the pinned revision with sha256
- [x] One GitHub release per repo, tagged `v1.0.0`

## Releasing a new version later

1. Train, then compare against the shipped model with `scripts/rank_sim.py` and the notebook's comparison table.
2. Quantize: `~/build/llama.cpp/build/bin/llama-quantize <f16>.gguf models/Emojet-vN-Q4_K_M.gguf Q4_K_M`.
3. Push the weights to Hugging Face (git-lfs), note the new commit sha.
4. Update `emojet-ai/shared/models.json` (new entry first, pinned URL + sha256) and `updates.json`.
5. Tag and release both repos. Existing installs pick the model up through "Check for updates".
