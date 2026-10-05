# Public release — decision and plan

**Status: decided, not started.** Nothing is published yet. Do the release work later, when the owner says go.

## Decision (owner, 2026-10-05)

- The model is released as **Emojet**, directly, at **https://huggingface.co/subhadeeproy3902/emojet**.
- The first public version is **Emojet v1**: the model trained on real data (internally called "v2" / Kaggle kernel `emojet-v2-real`) becomes public **v1**.
- The earlier internal models are **not** released: internal Emojet-v1 (`emojet-model-train`, `emojet-v1-resume`), v1-mini and v2-pre (`emojet-v2`). They stay development checkpoints.
- Goal: installable like any other open-weight model (Python `transformers`, llama.cpp GGUF, phones, desktop), fully open source.

## Name map (internal → public)

| Internal name | Kaggle kernel | Public |
|---|---|---|
| Emojet-v1 (46 tones, 4 styles) | `emojet-model-train`, `emojet-v1-resume` | not released |
| Emojet-v1-mini | `emojet-v1-mini`, `emojet-v1-mini-final` | not released |
| Emojet-v2-pre (from v1, topic-in-chat data) | `emojet-v2` | not released |
| **Emojet-v2 (real data, 65 tones, 8 styles, 6 axes)** | `emojet-v2-real` | **Emojet v1** → `subhadeeproy3902/emojet` |
| a distilled small version, only if it matches the full one on every test | (later) | a variant in the same repo or `emojet-small` |

From the release on, public version numbers count from v1. Internal file names in the app (`models.json`, `Emojet-v2-…gguf`) are renamed to the public ones at release time.

## Licence

- Weights: **CC BY-NC-SA 4.0** (free and open, no commercial use), because part of the real training data is non-commercial (e.g. BanglishRev, ESConv, MentalManip). Base model Laya (`convaiinnovations/laya`) is Apache-2.0: keep its licence text and credit.
- Code (this repo's scripts, the app): Apache-2.0.
- Model card lists every dataset with its licence and credit (see `training/real_data.py`, `training/tone_data.py`, `training/build_notebook.py`).

## What the Hugging Face repo will contain

1. `model.safetensors`, `config.json`, tokenizer files: standard `ModernBertForSequenceClassification`, works with stock `transformers` (`pipeline("text-classification", model="subhadeeproy3902/emojet", top_k=None)`).
2. `labels.json`: the 2,002 outputs in order (1,923 emojis, 65 tones, 8 styles, 6 axes) and how to read them.
3. GGUF files for llama.cpp: full precision + Q4_K_M (phone, ~274 MB). Needs `engine/llama.cpp-patches/0001-server-embedding-return-all-classifier-scores.patch` until it is upstream (proposing it to llama.cpp is a public action: ask the owner first).
4. Model card: what it does, input format (`replying to: …\nmy message: …`), results per language on held-out real messages and the chat cases, examples, data sources and licences, limitations and bias notes (real social-media data includes political and offensive text).
5. Optional later: a small `emojet` Python package / CLI (`pip install emojet`), free on PyPI.

## Before publishing (checklist)

- [ ] Emojet-v2 (real) beats internal v1 and v2-pre on every test (real held-out per language, chat cases, tone/style).
- [ ] Owner reviews the model card and files.
- [ ] Owner creates the Hugging Face account/org `subhadeeproy3902` (free) and a **write token**, stored in `emojet/.env` as `HF_TOKEN=…` (never committed).
- [ ] App (`../emojet-ai`) points `models.json` at the Hugging Face file (pinned revision + sha256), label file `tones.json` v2.
- [ ] Single Kaggle notebook (definition, weights, live testing, evaluation, full pipeline) and a Kaggle collection with all EmoJet notebooks.
