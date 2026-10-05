# Emojet (model) — context for Claude Code

Emojet = open model that reads a whole message (+ the one it replies to) and scores 1,923 emojis + 46 tones + 4 styles + 4 tone dimensions in one pass. Versions: Emojet-v1, v2… (fine-tunes of Laya, Apache-2.0). The app that uses it is the sibling repo `../emojet-ai`. Owner: Subhadeep Roy (subhadeeproy3902), Linux beginner — explain in plain words.

## Rules
- Commit and push after every meaningful change (owner's standing instruction), with the Co-Authored-By Claude trailer.
- Never commit `.env` (Kaggle token: `KAGGLE_API_TOKEN=KGAT_…`, `KAGGLE_USERNAME=subhadeep3902`), `models/`, `data/`, `*.gguf`, `*.safetensors`.
- Don't publish weights (GitHub release / Hugging Face) until the owner explicitly says so.
- **Release plan: `docs/RELEASE.md`.** Public name **Emojet**, repo https://huggingface.co/subhadeeproy3902/emojet; the real-data model (internal "v2", kernel `emojet-v2-real`) becomes public **v1**. Earlier internal models are not released.
- Laya only as the base; $0 (Kaggle free GPU for training).

## Layout
- `training/build_notebook.py <commit> [--smoke] [--mini] [--resume]` builds the Kaggle notebook (data embedded); push with `python3 scripts/kaggle.py push <ipynb> <slug> --gpu --internet [--source=subhadeep3902/emojet-model-train]`; fetch with `scripts/kaggle.py output <slug> <dest> --only=…` (resumable).
- Kaggle notebooks: `emojet-model-train` (v1, 3 epochs), `emojet-v1-mini` (distilled, 14 layers), `emojet-v1-resume` (v1 + chat-object data, 1 epoch), `emojet-v2` (v2-pre), `emojet-data` (real messages, CPU), `emojet-tones` (human-labelled tone sets, CPU), `emojet-annotate` (Qwen3.6-35B-A3B labeller, 2xT4), `emojet-v2-real` (real-data model = public v1). Links: https://www.kaggle.com/code/subhadeep3902/<slug>. Give the owner the links; no polling watchers (owner pings when done).
- `labels/` = label space (emoji-catalog, tones; order = model outputs). `engine/llama.cpp-patches/` must be applied (`scripts/build-engine.sh`).
- Test: `MODEL=models/Emojet-v1-Q4_K_M.gguf bash scripts/dev-server.sh &` then `python3 scripts/test-emojet.py [--reference data/kaggle/<run>/reference.json]`.
- Quantization (2026-10-05): Q4_K_M best size/drift balance for v1 (273 MB); Q4_0 drifts too much for this classifier.
