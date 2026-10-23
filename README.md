<div align="center">

<img src="https://raw.githubusercontent.com/subhadeeproy3902/emojet-ai/main/assets/brand/emojet-logo-256.png" width="140" alt="Emojet">

# Emojet

**The model that reads your whole sentence and knows which emoji you meant.**

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-subhadeep3902%2Femojet-ffcc4d)](https://huggingface.co/subhadeep3902/emojet)
[![Phone file](https://img.shields.io/badge/phone_file-274_MB-blue)](https://huggingface.co/subhadeep3902/emojet/blob/main/Emojet-v1-Q4_K_M.gguf)
[![Code licence](https://img.shields.io/badge/code-Apache--2.0-green)](LICENSE)
[![Weights licence](https://img.shields.io/badge/weights-CC_BY--NC--SA_4.0-lightgrey)](https://huggingface.co/subhadeep3902/emojet)
[![Languages](https://img.shields.io/badge/EN_%C2%B7_HI_%C2%B7_BN-%2B_Hinglish_%C2%B7_Banglish-orange)](#languages)
[![New emoji watch](https://github.com/subhadeeproy3902/emojet/actions/workflows/emoji-watch.yml/badge.svg)](../../actions/workflows/emoji-watch.yml)

[**Model card**](https://huggingface.co/subhadeep3902/emojet) · [**The app**](https://github.com/subhadeeproy3902/emojet-ai) · [Results](#results) · [How it was built](#how-it-was-built) · [Train it yourself](#train-it-yourself) · [Contributing](CONTRIBUTING.md)

</div>

---

Emojet reads a **whole chat message** — and the message it replies to — and, in **one forward pass**, scores every
emoji in Unicode 18.0, the **tone of voice** it is written in, and the **way the person communicates**.

```
                      ┌──────────────────────────────────────────┐
 "ghum pacche khub"   │  Laya encoder (ModernBERT-large, 28L)    │   😴 💤 🛌 🥱 😪 🌙
 ───────────────────▶ │  + one 1024 → 2002 classification head   │ ▶ tired · resigned · casual
   (+ the reply)      └──────────────────────────────────────────┘   style: passive
                              398 M params · 274 MB at 4-bit
```

| Group | Outputs | What it gives you |
|---|---|---|
| **emojis** | 1,923 | every distinct emoji in Unicode 18.0 (the app applies skin tones) |
| **tones** | 65 | `sarcastic`, `passive-aggressive`, `condescending`, `flirty`, `resigned`, `obsequious`, … |
| **styles** | 8 | assertive · passive · aggressive · passive-aggressive · manipulative · collaborative · transactional · avoidant |
| **axes** | 6 | funny↔serious, formal↔casual, respectful↔irreverent, enthusiastic↔matter-of-fact, high↔low context, linear↔circular |

Fine-tuned from [**Laya**](https://huggingface.co/convaiinnovations/laya) (Apache-2.0). Trained on **real chat
messages**, not synthetic examples. Runs offline: about **180 ms** on a laptop CPU, **350–850 ms** on a mid-range phone.

## Why not just match keywords

Keyword matching looks at the last word. The meaning is in the sentence.

| Message | Keyword matching | Emojet |
|---|---|---|
| `my program finally works` | 💻 | 🥳 🥲 😭 🥹 |
| `oh great, another monday meeting` | 📅 | 🙄 🙃 😒 😮‍💨 |
| `ghum pacche khub` *(Banglish: very sleepy)* | — | 😴 💤 🛌 🥱 |
| `kal paper hai aur kuch nahi padha` *(Hinglish: exam tomorrow, studied nothing)* | 📄 | 📚 😭 😩 😰 |

## Languages

English, Hindi and Bengali — **including the way people actually type them**: Hinglish and Banglish in Latin letters,
native script, and the spelling variation in between (`pacche` / `pachhe` / `pache`, `bhalobashi` / `valobasi`).

## Results

Held-out data the model never trained on. `hit@6` = one of the six suggested emojis is one the writer really used —
the number that matters for a six-emoji strip.

### Real messages · 2,000 held out per language

| | English | Hindi | Bengali |
|---|---|---|---|
| top-1 | 25.6% | 26.8% | 24.4% |
| **hit@6** | **52.4%** | **55.4%** | **61.4%** |

Random choice among 1,923 emojis is 0.05%. Two people writing the same sentence rarely choose the same emoji, so
~25% exact agreement is a strong result rather than a weak one.

### Chat, slang, tone

| Test | Score |
|---|---|
| Chat cases (40 WhatsApp-style messages), right emoji in top 3 | 31/40 |
| Slang & everyday Banglish/Hinglish probes (72), hit@6 | 41/72 |
| Lookalike traps (8) that must **not** get toilet/swear emojis | 6/8 |
| Emoji concept cases (24), hit@6 | 23/24 |
| Emojinize multilingual test set, hit@6 | 99.3% |
| Tone in top 3 (25 cases; untrained baseline 18/25) | 23/25 |
| Communication style (13 cases; untrained baseline 6/13) | 11/13 |
| Hinglish sarcasm, held-out accuracy | 81.7% |

### What it still gets wrong

- **Crude Bengali slang.** `Haga pacche` returns 😂 instead of 💩🚽 — only ~112 such messages exist in 2.9 M real
  ones, and in most of them the writer typed 😂. Words with more data (`ghum`, 700) work.
- **Tone on very short romanized Bengali.** `Brishti porche baire` (*it's raining*) scores 0.93 on "angry".
- **Hindi slang trails Bengali** (8/19 vs 25/44 on the probes).

Full history, including the rounds that regressed and why, is in [`docs/TRAINING.md`](docs/TRAINING.md).

## Use it

```bash
pip install transformers torch
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("subhadeep3902/emojet")
model = AutoModelForSequenceClassification.from_pretrained("subhadeep3902/emojet").eval()

def emojet(draft, reply=None, k=6):
    text = (f"replying to: {reply.strip()}\n" if reply else "") + f"my message: {draft.strip()}"
    with torch.no_grad():
        logits = model(**tok(text, return_tensors="pt", truncation=True, max_length=256)).logits[0]
    return [model.config.id2label[i] for i in logits[:1923].topk(k).indices.tolist()]

print(emojet("my program finally works"))
```

On a phone, or a laptop without Python, use the 4-bit GGUF with llama.cpp — see the
[model card](https://huggingface.co/subhadeep3902/emojet#quick-start).

## How it was built

Three stages, ~40 GPU-hours in total on a free Kaggle T4. Every stage is a notebook this repo generates.

| Stage | What it teaches | Data |
|---|---|---|
| **1. Concepts** | an emoji means a thing | CLDR names and keywords, emojinize spans, things mentioned in casual chat |
| **2. Real messages** | what people actually send | 2.9 M public messages (EN/HI/BN) where the writer used an emoji — the emoji is stripped out and becomes the label |
| **3. Tone, style, slang** | how it is said | human-labelled tone/sarcasm/politeness/toxicity sets mapped onto the 65-tone taxonomy, ~88 K messages labelled by an open LLM, and a 146-entry chat lexicon of Banglish/Hinglish words |

Two techniques carry most of the quality:

- **Balanced softmax** (τ = 0.5) stops 😂❤️👍 from swallowing the long tail — without it the model suggests the same
  six popular emojis for everything.
- **Weight averaging** (WiSE-FT style): the released weights are a blend of the real-message model and the slang
  model, at the ratio that kept the slang gains with no statistically significant loss on real messages
  (paired McNemar test per language).

## Train it yourself

Everything runs on free hardware. No paid API, no private data.

```bash
git clone https://github.com/subhadeeproy3902/emojet && cd emojet
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round3     # generates the Kaggle notebook
python3 scripts/kaggle.py push training/kaggle/emojet_v2_round3.ipynb my-run --gpu --internet --source=...
```

The run measures the starting model, measures again halfway, and ends with a table that flags anything that got
**worse** than the shipped model. [`docs/TRAINING.md`](docs/TRAINING.md) has the full pipeline and every round that
was actually run — including the ones that regressed and what caused it.

## Repo map

| Path | What is in it |
|---|---|
| `labels/` | the emoji catalog (1,923) and the tone/style/axis taxonomy |
| `training/` | data jobs, the notebook builders, the chat lexicon and its matcher |
| `training/data/eval_sets.json` | every test the model is judged on |
| `scripts/` | Kaggle client, the ranking simulator, quantization, the emoji-catalog builder |
| `docs/TRAINING.md` | the full training history, round by round |
| `docs/RESEARCH.md` | what was tried and rejected |
| `engine/` | the llama.cpp patch that returns all classifier scores |

## Staying current with Unicode

A [monthly GitHub Action](.github/workflows/emoji-watch.yml) rebuilds the catalog from unicode.org and CLDR. If new
emoji appear it opens a pull request with the updated catalog and an issue explaining that the output layer has to
grow — the training code detects a larger label set, keeps every existing output row and initialises only the new ones.

## Licence

- **Code** (this repo): [Apache-2.0](LICENSE).
- **Weights** ([on Hugging Face](https://huggingface.co/subhadeep3902/emojet)): CC BY-NC-SA 4.0 — free and open,
  non-commercial, because part of the training data is non-commercial.
- Base model [Laya](https://huggingface.co/convaiinnovations/laya): Apache-2.0.
- Dataset licences and credits: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Credits

[Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations · [llama.cpp](https://github.com/ggml-org/llama.cpp)
· Unicode CLDR · and every dataset author listed in the notices.

<div align="center">

**[Get the app →](https://github.com/subhadeeproy3902/emojet-ai)**

</div>
