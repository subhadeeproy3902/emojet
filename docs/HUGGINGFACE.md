# Hugging Face — what EmoJet needs from it (and what it doesn't)

**Short answer: nothing that costs money, and for now not even an account.** The Laya models are public, not gated, Apache-2.0. Anyone (you, the app, the extension) can download them anonymously.

## What needs what

| Task | HF account? | Token? | Cost | When |
|---|---|---|---|---|
| Download Laya for development | no | no | $0 | now (done: `bash scripts/get-model.sh`) |
| App / extension downloads the model on first run | no | no | $0 | Phase 2 / 3 |
| Publish our own smaller model file (e.g. 4-bit quant) | **yes, free** | **yes, write token, only on your laptop** | $0 | only if Phase 1 picks a quant that ggml-org doesn't publish |
| Fine-tune Laya on emoji data | free account helps | — | HF's paid GPUs are **not** needed: train on your RTX 2050 or free Colab | optional, much later |
| Hugging Face "Inference API" / Endpoints | — | — | not used | never — it would be a server, the opposite of the design |

## The exact files we use

| | Value |
|---|---|
| Repo | [`ggml-org/Laya-GGUF`](https://huggingface.co/ggml-org/Laya-GGUF) (converted from [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)) |
| Pinned revision | `22265007700297ba9e128297e82540cf28c5d7d4` |
| `Laya-Q8_0.gguf` | 449,397,600 bytes, sha256 `c06528c5746d3bb8baa72a27938be95abbfd0b226f8471e8a9e365ed0bb066d2` |
| `Laya-BF16.gguf` | 844,026,720 bytes, sha256 `edeea94939cd1575c1f984c28982170c0e47caab66e4b80993fed992519573b4` |
| Download URL | `https://huggingface.co/ggml-org/Laya-GGUF/resolve/<revision>/<file>` |

Only the main `laya` checkpoint (English, 421M) has a GGUF. The `multilingual` (322M, mmBERT) and `typed-decisions` checkpoints exist only as safetensors in the source repo; converting them is possible later with llama.cpp's converter if Hinglish/other languages need it.

## Rules the apps follow when downloading

1. **Pinned revision, not `main`** — an upstream change can't silently change or break installed apps.
2. **sha256 check** after download; a mismatch deletes the file and retries.
3. **Resumable** download with a progress bar, Wi-Fi-only by default on Android (it's ~0.25–0.45 GB).
4. **Fallback order**: Hugging Face → GitHub Releases mirror (our repo, free, up to 2 GiB per file) → "Import a .gguf file" button (for offline installs or if both are blocked).
5. Hugging Face rate-limits anonymous downloads per IP. Irrelevant for you and friends; the GitHub mirror covers a sudden spike.

Browser note: the HF link answers with a redirect to its CDN (currently `us.aws.cdn.hf.co`). The extension declares host permissions for `https://huggingface.co/*` and `https://*.hf.co/*`, so CORS is not an issue.

## Licence duties (Apache-2.0) when we re-host or bundle the model

Downloading it from HF at runtime needs nothing special. If we **mirror** it (GitHub Releases) or publish a **quantized copy**, we must:

- ship the Apache-2.0 licence text with it,
- credit the source (`convaiinnovations/laya`, converted by `ggml-org`),
- say what we changed (e.g. "re-quantized to Q4_K_M with llama.cpp <commit>"),
- keep any upstream `NOTICE` file (none exists at the time of writing).

All of this goes in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and in the model card.

## If we ever publish our own model file — your steps (≈5 minutes)

1. Create a free account at huggingface.co and verify the email.
2. Settings → Access Tokens → **Create new token** → fine-grained, **write access only to the repo we'll create** (or type "Write"). Name it `emojet-upload`. Treat it like a password.
3. Install the official CLI (from the Arch repo): `sudo pacman -S python-huggingface-hub`
4. Log in on this laptop: `! hf auth login` (paste the token; it's saved in `~/.cache/huggingface/token`, never in the repo).
5. Tell Claude the account name. Claude creates `<you>/emojet-laya-gguf`, uploads the file with a model card (`base_model: convaiinnovations/laya`, `license: apache-2.0`), and pins the new revision + sha256 in the apps.
6. Optional: delete the token afterwards; create a new one when needed.
