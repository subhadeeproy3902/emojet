# Phase T — Train "Laya-Emoji"

**Goal:** one model that reads the **whole message** (and the message you're replying to) and, in a single fast pass, scores **every emoji** — all 1,923 distinct emojis of Unicode 18.0 (3,963 including skin-tone variants) — **and the tone of voice** (46 tones, 4 communication styles, 4 tone dimensions). Still Laya: it starts from Laya's own encoder weights (Apache-2.0) and is fine-tuned for emojis.

## Why the current setup can't do this

- Laya's built-in "choice" question holds at most 255 options inside a 192-token budget. 1,923 emojis can't fit, even as words. That's why Phase 0 uses ~25 mood words.
- Moods cover **feelings** well (18/23 right) but not **things**: "landed in Goa, beach time" should offer 🏖️🌴✈️, "pizza tonight?" 🍕, "India won!" 🇮🇳🏏. The mood design can only reach the emojis we hand-place in 25 rows.

## The design

```
reply context + whole draft (≤ 64 tokens, emojis stripped)
        │
        ▼
Laya encoder (ModernBERT-large, 28 layers, initialised from convaiinnovations/laya)
        │  [CLS] vector, 1024 numbers
        ▼
new emoji head: 1024 → 1,923 scores (one per base emoji, sigmoid = multi-label)
        │
        ▼
top 6 → apply skin-tone preference → personal boost → overlay
```

- **Whole sentence, always.** The input is the full draft + reply context; there is no per-word lookup in the model path. A message like "not bad at all, honestly loved it" must not trigger 👎 because of "bad".
- **Multi-label**: a message can deserve 🎉 and 🎂 and 🥳 at the same time — trained with one yes/no output per emoji, not "pick one".
- **Faster than today**: the prompt is only the message (~20–40 tokens) instead of question + 25 options (~80 tokens). Expected ~2× faster per suggestion, while covering 77× more emojis.
- **Format**: Hugging Face `ModernBertForSequenceClassification` with 1,923 labels → `convert_hf_to_gguf.py` (llama.cpp supports this class, labels are stored in the file) → `llama-quantize` Q4_0. Head adds ~2M parameters (~1–4 MB).

## Tone of voice (owner's addition, 2026-10-04)

The same pass also reads **how** the message is said. Taxonomy: [`shared/tones.json`](../shared/tones.json) (sources in RESEARCH.md §5):

- **4 communication styles** (pick one): assertive · passive · aggressive · passive-aggressive.
- **46 tones** (any number can apply), in 5 families — intellectual & analytical (sarcastic, neutral, cynical, witty, analytical, curious, skeptical, thoughtful, pessimistic, serious), positive & engaging (empathetic, enthusiastic, encouraging, playful, humorous, joyful, loving, appreciative, admiring, optimistic, friendly, confident, flirty, nostalgic), defensive & hostile (condescending, defensive, apathetic, accusatory, aggressive, angry, disapproving, bitter, demanding, rude), formal & submissive (authoritative, formal, respectful, apologetic, tentative, urgent), vulnerable & emotional (sad, anxious, surprised, disappointed, confused, embarrassed). Each has aliases (e.g. condescending = patronizing, arrogant, smug).
- **4 tone dimensions** (0–1 sliders, Nielsen Norman Group): funny↔serious · formal↔casual · respectful↔irreverent · enthusiastic↔matter-of-fact.

### How it fits in the model — zero extra cost

All outputs share the same encoder pass; they are just more numbers at the end:

```
[CLS] vector ──► one output layer, 1,977 numbers
                 ├─ 1,923  emoji scores      (sigmoid each)
                 ├─    46  tone scores       (sigmoid each)
                 ├─     4  style scores      (softmax)
                 └─     4  dimension scores  (sigmoid)
```

One output layer keeps it compatible with llama.cpp's classifier format (it stores one projection + labels); the app splits the numbers into the four groups. Training uses one loss per group, added together (multi-task), so learning tone also sharpens the emoji choice (a sarcastic "great job" should get 🙃👏, not 🥳).

### What the tone outputs are used for

1. **Tone-matched emojis** — final emoji score = emoji head + a small boost for the emoji rows of the detected tones (`tones.json` → `emojis`).
2. **Tone chip** (proposed) — a tiny label on the strip: "playful", "sounds passive-aggressive", like Grammarly's tone detector.
3. **Tone check before sending** (proposed, off by default) — if the draft scores high on aggressive / condescending / passive-aggressive / accusatory, show a gentle "😬 may sound passive-aggressive".
4. **Their tone too** — the same model on the replied-to message ("they sound upset" → 🫂 rises).

### Baseline: untrained Laya (2026-10-04, `scripts/test-tones.py`, 25 labelled messages)

| | Result |
|---|---|
| Tone, first pick / top 3 | 14/25 · 18/25 |
| Communication style | 6/13 (mostly answers "passive") |
| Speed | ~1.15 s for both questions (46 tone options ≈ 170 tokens per question) |

Right on explicit tones (apologetic, empathetic, anxious, surprised, encouraging, playful, appreciative, flirty→admiring). Wrong on implicit ones: sarcasm ("my favourite thing ever" → joyful), passive-aggressive ("since you always know best" → neutral), condescending, cynical, accusatory, defensive. → Untrained Laya is a usable *teacher* only for explicit tones; the subtle ones need the real datasets below.

## Training data (all free, licences allow training)

| Source | What it teaches | Size | Licence |
|---|---|---|---|
| [`yagizgencer/emojinize-multilingual`](https://huggingface.co/datasets/yagizgencer/emojinize-multilingual) | **things/concepts** in full sentences → emojis (🇮🇹 🚀 ⚡ 🍕 …), 14 languages incl. Hindi | 108K sentences, ~430K emoji spans | CC-BY-4.0 (credit required) |
| [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | **feelings** in real chat-style comments (27 emotions) → mapped to emoji sets | 58K labelled comments | Apache-2.0 |
| GoEmotions (same) → tone map | 27 emotions map onto ~20 tones (admiration→admiring, amusement→humorous, annoyance→angry, caring→empathetic, gratitude→appreciative, remorse→apologetic, nervousness→anxious, disapproval→disapproving, …) | 58K | Apache-2.0 |
| [`CreativeLang/SARC_Sarcasm`](https://huggingface.co/datasets/CreativeLang/SARC_Sarcasm) | **sarcastic** (self-labelled Reddit "/s") | 1M+ (sample) | CC-BY-2.0 |
| [`Cleanlab/stanford-politeness`](https://huggingface.co/datasets/Cleanlab/stanford-politeness) | respectful ↔ rude, demanding | ~11K | MIT (orig. CC-BY-4.0) |
| [`osyvokon/pavlick-formality-scores`](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) | formal ↔ casual dimension | ~11K | CC-BY-3.0 |
| [`tasksource/jigsaw_toxicity`](https://huggingface.co/datasets/tasksource/jigsaw_toxicity) | aggressive, rude, accusatory (insult/threat labels) | 100K+ | Apache-2.0 |
| Laya (untrained) as teacher | explicit tones + moods on extra chat sentences → soft labels | as many as we generate | own |
| Hand-written examples | the 4 communication styles, condescending, defensive, cynical, passive-aggressive (no good free dataset exists) | a few hundred | yours / ours — see MANUAL-WORK §B |
| Unicode CLDR names + keywords | a few template sentences for every emoji, so none of the 1,923 is never seen | ~6K | Unicode-3.0 |
| **Your ~50 messages** | real target style (English + Hinglish) | 50+ | yours — used as the **test set**, not for training |

Labels per sentence = the set of emojis attached to it. Emojis already inside the input text are removed (the model must predict them, not copy them). Rare emojis get weighted up so 😂 doesn't win every time.

Licence filter: datasets marked non-commercial or no-derivatives (e.g. EmpatheticDialogues CC-BY-NC, ToneStyle TST2K CC-BY-NC-ND) are **not used**, so the model and app can stay MIT/Apache and free for anyone. Datasets with no stated licence are skipped too.

Notes: emojinize was annotated by GPT models (stated in its card) and is encyclopedic, not chatty — GoEmotions + Laya-teacher data balance the tone. Credits go in `THIRD_PARTY_NOTICES.md`.

## Where to train (both $0) — owner decides

| | A. This laptop (RTX 2050, 4 GB) | B. Kaggle free GPU (2× T4, 30 h/week) |
|---|---|---|
| Setup | `sudo pacman -S python-pytorch-cuda python-transformers python-safetensors python-datasets` (~5 GB) | free Kaggle account + phone verification; Claude writes the notebook, you press Run |
| Method | 4 GB is too small for full fine-tuning → LoRA adapters + head, or freeze lower layers | full fine-tuning |
| Time per run | ~1–3 h, laptop GPU hot and loud | ~30–60 min |
| Who runs it | Claude, end to end | you start it; Claude prepares everything |
| Fits "low temperature" goal | no | yes |

Recommendation: **data prep + evaluation on the laptop (Claude), training on Kaggle (B)**; a small local pilot (A, frozen encoder + head only, ~minutes) first to prove the pipeline before spending Kaggle hours.

## Notebooks on Kaggle (private, under the owner's account)

| Notebook | What | Link |
|---|---|---|
| `emojet-laya-check` | step 1: Laya vs llama.cpp, Laya-Emoji shape, T4 speed | https://www.kaggle.com/code/subhadeep3902/emojet-laya-check |
| `emojet-inspect-data` | layout of the Excel sarcasm files | https://www.kaggle.com/code/subhadeep3902/emojet-inspect-data |
| `emojet-laya-emoji-smoke` | the training notebook on 4,000 examples, 1 epoch — catches bugs in minutes | https://www.kaggle.com/code/subhadeep3902/emojet-laya-emoji-smoke |
| `emojet-model-train` (pushed as `emojet-laya-emoji-train`, renamed on Kaggle) | **the full training run** | https://www.kaggle.com/code/subhadeep3902/emojet-model-train |

Built from `training/build_notebook.py <commit> [--smoke]` (data files are fetched from that exact repo commit) and sent with `python3 scripts/kaggle.py push … --gpu --internet`; results come back with `python3 scripts/kaggle.py wait <slug>` into `data/kaggle/<slug>/` (git-ignored). Open a notebook on Kaggle → **Output / Logs** to see every number; **Copy & Edit** to run the "Try it" cell with your own sentences.

Model input text (training and every app must build exactly this): `replying to: <reply>\nmy message: <draft>` — first line only when replying.

## Steps

1. ✅ [Claude] **Done 2026-10-05 on Kaggle** (`training/kaggle/laya_check.py`, private notebook `subhadeep3902/emojet-laya-check`, T4 16 GB, 83 s):
   - Laya's own PyTorch code vs our llama.cpp BF16 engine on mood / tone / style / yes-no / score questions: **20/20 identical answers**, identical token counts, probabilities within ~0.002 (worst 0.064 on a near-tie, fp16 vs bf16).
   - Laya's encoder (170 tensors, prefix `encoder.`) loads into `ModernBertForSequenceClassification` with 1,977 outputs: **no missing / unexpected keys, hidden states identical (max diff 0.0)**. Model = 397.9 M parameters. Laya's own decision head (`head`, `scorer`, `type_emb`, `act_head`) is not used.
   - Full fine-tuning fits on the free T4: 107 samples/s at batch 64 × 48 tokens (11.1 GB peak), 70 samples/s at 32 × 64 (9.7 GB) → **~31–48 min per epoch of 200K samples**.
   - Kaggle stack: torch 2.11 (CUDA 12.8), transformers 5.16.
2. ✅ [Claude] Dataset is built inside the notebook from the sources above + `training/data/` (seeds, GoEmotions map, eval sets). Note: the Hugging Face SARC copy has no sarcasm labels, so sarcasm comes from the CC-BY-4.0 Hinglish/English headlines (BM2002) + Apache-2.0 irony tweets.
3. [Claude] Pilot: frozen encoder + head only, on the laptop CPU/GPU. Measure hit@6 on validation and on the 23 Phase-0 samples (mapped to acceptable emoji sets).
4. [Claude + You] Full fine-tune (Kaggle or local LoRA).
5. [Claude] Convert → GGUF → Q4_0; check llama.cpp gives the same scores as PyTorch; latency on the laptop.
6. [Claude] A/B against the Phase-0 mood model on your test set. Ship Laya-Emoji only if it wins.
7. [Claude] Publish the model file (GitHub Release asset, optional Hugging Face repo — see HUGGINGFACE.md) with a model card crediting Laya, emojinize, GoEmotions and CLDR.

## Metrics

- **hit@6**: at least one emoji you'd accept is among the 6 shown (main metric). Target ≥ 85% on your messages.
- **mAP over emojis**, coverage (how many of the 1,923 emojis ever reach a top-6 on validation).
- **Tone**: top-3 hit rate on `scripts/test-tones.py` + your messages (baseline untrained Laya: 18/25), per-tone F1 for the subtle ones (sarcastic, passive-aggressive, condescending, cynical, accusatory, defensive). **Style** accuracy (baseline 6/13).
- Latency on laptop and phone; model size.
