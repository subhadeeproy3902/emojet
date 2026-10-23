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

## Round 2 (prepared 2026-10-06, run when the Kaggle GPU quota allows ~13 h)

Continues the real-data model (public "Emojet v1" candidate) with slang/swearing and more Banglish/Hinglish.
Run in this order; each step needs the previous one's output. Links: https://www.kaggle.com/code/subhadeep3902/<slug>

| # | Kaggle notebook | Machine | Time | What |
|---|---|---|---|---|
| 1 | `emojet-slang` | CPU | ~20–40 min | slang/swear lexicons (EN, Hindi, Hinglish, Bengali, Banglish) + real messages using them + human labels of slang/abuse sets |
| 2 | `emojet-annotate-2` | 2×T4 | ~3 h | tone/style/emoji labels for the slang messages (Banglish first) + 14k more Bengali/Hindi messages; round-1 texts skipped |
| 3 | `emojet-v2-round2` | T4 | ~9.5 h | continue from `emojet-v2-real`: unseen half of the data, slang ×3, Bengali ×3, Hindi ×2, LR 1e-5 / 8e-5 |

```bash
# 1
python3 training/build_slang.py && python3 scripts/kaggle.py push training/kaggle/emojet_slang.py emojet-slang --internet --source=subhadeep3902/emojet-data
# 2 (after 1)
python3 training/build_annotate.py --round2 && python3 scripts/kaggle.py push training/kaggle/emojet_annotate_round2.py emojet-annotate-2 --gpu --internet \
    --source=subhadeep3902/emojet-data --source=subhadeep3902/emojet-slang --source=subhadeep3902/emojet-annotate
# 3 (after 2)
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round2 && python3 scripts/kaggle.py push training/kaggle/emojet_v2_round2.ipynb emojet-v2-round2 --gpu --internet \
    --source=subhadeep3902/emojet-v2-real --source=subhadeep3902/emojet-data --source=subhadeep3902/emojet-tones \
    --source=subhadeep3902/emojet-slang --source=subhadeep3902/emojet-annotate --source=subhadeep3902/emojet-annotate-2
```
Smoke-test step 1 first (`--smoke`, slug `emojet-slang-smoke`, a few minutes, CPU). Do not attach `emojet-v2` to step 3 (the starting model is found by its `emojet.json`).

### Round 2 result (2026-10-13)

Ran for 1 epoch (50,803 steps), hit the ~10 h time budget as planned. `models/Emojet-v2round2-Q4_K_M.gguf`
(274 MB, same size as v2-real's), registered in `emojet-ai/shared/models.json` as `emojet-v2round2`, now first
in the list (app default). Compared with v2-real:

| Metric | v2-real | v2-round2 |
|---|---|---|
| English real top-1 / hit@6 | 24.5% / 51.2% | 24.7% / 52.9% |
| Hindi real top-1 / hit@6 | 24.3% / 54.6% | **26.9%** / 55.4% |
| Bengali real top-1 / hit@6 | 25.9% / 62.1% | 25.3% / 62.3% |
| Chat cases top-1 / top-3 | 22/40 / 26/40 | 22/40 / **30/40** |
| Hinglish sarcasm acc | 0.826 | 0.824 |

Net improvement, not a regression. **But the one case round 2 specifically targeted still fails**: "Haga
pacche" (expected toilet/poop emojis) still returns laughing emojis, same as before round 2, despite 400
real/mined examples of "haga" in the slang data and the document-frequency filter keeping it. Likely cause:
one epoch isn't enough exposure to a few hundred examples inside a 50k-message training run; would need
either more epochs on the slang data specifically, or oversampling it harder than ×3. Not re-run yet — flagged
for the owner to decide whether it's worth another round before release.

## Round 3 (2026-10-22): slang and everyday Banglish/Hinglish, done properly

### Why round 2 didn't fix "Haga pacche"
1. **Half its slang data was junk.** Terms from the Bengali/Hindi lists were matched as raw substrings: "haga" fired inside
   "bhagao"/"Bhagavad", "nti"/"pla"/"hol"/"chen" inside ordinary words. 32,892 of 67,230 such messages had no slang, and they
   were weighted ×3. Their language came from the word list, not the message (Hindi tweets tagged Bengali).
2. **Typed emojis threw the labeller's away.** Training used `gold or silver`: a real "হাগু 😂" message taught 😂 only, never 💩.
3. **The labeller was never told what slang means**, and Qwen does not know Banglish "haga".
4. **Everyday Banglish is missing too, not just slang.** Our Bengali corpus is reviews and YouTube comments: "ghum pacche",
   "brishti porche", "cha khabi" barely occur (about 200 poop-related messages in 2.9 M).

Baseline (round-2 phone model, `scripts/rank_sim.py`): slang/everyday probes **29/72** (bn 16/44, hi 7/19, en 6/9),
lookalikes clean 5/8, chat cases 22/40 top-1, 29/40 top-3.

### What round 3 changes
- `training/data/chat_lexicon.json`: 146 entries (Bengali/Banglish, Hindi/Hinglish, English internet slang), ~1,300
  spellings in Latin and native script. Each has a meaning and, for things/states, emojis. Kinds: topic (always applies),
  reaction (short messages only), swear and address (meaning only). Ambiguous spellings were removed after a scan of the real
  corpus (বাজে = o'clock, সেই = that, cha = want, puja = a name, পরীক্ষা করে = testing a product, কাজ করছে = it works,
  khuda = God, nite paren = you can buy, based on, ...).
- `training/chatlex.py`: one matcher used by the miner, the labeller and training. Whole words, spelling folding for
  romanized text (pacche/pachhe/pache, hagaa, valobasi/bhalobashi), suffixes only from real endings (বৃষ্টিতে, brishtite;
  not শীতল), exact match for swears (folding would turn "chhod do" into a swear), and per-language gating.
- `slang_data.py`: mines with the matcher; language from the message; external profanity lists filtered by document
  frequency per language and script (0.1%; curated profanity lists exempt); asserts every target contains its word.
- `annotate.py --round3`: the meaning of each word goes into the prompt; round-robin over words so every word gets labels
  before time runs out; then up to 1.25 h of synthetic WhatsApp-style lines per lexicon entry (kept only if the matcher
  finds the word; stops itself if most lines miss it).
- `build_notebook.py --round3`: continue from round 2. One soft target per message: typed emojis 1.0, labeller 0.5, word
  meaning 0.4. Meaning emojis on every real message that contains a lexicon word. Round-2 "slang" weighting only if a
  lexicon word is really there. Synthetic lines ×2 at weight 0.5 (never a test message). Spelling variation on repeated
  Banglish/Hinglish copies. Old data replayed at 40-80%. Starting-model evaluation, halfway evaluation, final comparison
  table (↓ WORSE flags). An out-of-memory batch is skipped; any other crash still evaluates and saves.
- `training/data/eval_sets.json`: `slang_probes` (72: Banglish, Bengali script, Hinglish, Devanagari, English slang,
  misspellings, replies) and `slang_negatives` (8 lookalikes that must not get toilet/swear emojis). Test only.

### Run order (links: https://www.kaggle.com/code/subhadeep3902/<slug>)
| # | Kaggle notebook | Machine | Time |
|---|---|---|---|
| 1 | `emojet-slang-3` | CPU | ~30-45 min |
| 1b | `emojet-v2-round3-smoke` (catches code errors on existing data) | T4 | ~40 min |
| 2 | `emojet-annotate-3` | 2×T4 | ~4 h |
| 3 | `emojet-v2-round3` | T4 | ~10 h |

```bash
# 1 and 1b (independent)
python3 training/build_slang.py && python3 scripts/kaggle.py push training/kaggle/emojet_slang.py emojet-slang-3 --internet --source=subhadeep3902/emojet-data
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round3 --smoke && python3 scripts/kaggle.py push training/kaggle/emojet_v2_round3_smoke.ipynb emojet-v2-round3-smoke --gpu --internet \
    --source=subhadeep3902/emojet-v2-round2 --source=subhadeep3902/emojet-data --source=subhadeep3902/emojet-tones \
    --source=subhadeep3902/emojet-annotate --source=subhadeep3902/emojet-annotate-2
# 2 (after 1)
python3 training/build_annotate.py --round3 && python3 scripts/kaggle.py push training/kaggle/emojet_annotate_round3.py emojet-annotate-3 --gpu --internet \
    --source=subhadeep3902/emojet-data --source=subhadeep3902/emojet-slang-3 --source=subhadeep3902/emojet-annotate --source=subhadeep3902/emojet-annotate-2
# 3 (after 2 and a clean 1b)
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round3 && python3 scripts/kaggle.py push training/kaggle/emojet_v2_round3.ipynb emojet-v2-round3 --gpu --internet \
    --source=subhadeep3902/emojet-v2-round2 --source=subhadeep3902/emojet-data --source=subhadeep3902/emojet-tones --source=subhadeep3902/emojet-slang-3 \
    --source=subhadeep3902/emojet-annotate --source=subhadeep3902/emojet-annotate-2 --source=subhadeep3902/emojet-annotate-3
```
Attach only one model (`emojet-v2-round2`): the starting model is found by its `emojet.json`. Do not attach `emojet-slang`
(round 2) to steps 2-3: its targets are the junk ones. After step 3: download with `scripts/kaggle.py output` (now checks
the file size), quantize Q4_K_M, `scripts/rank_sim.py` against the baseline above, and ship only if nothing got worse.

### Round 3 result (2026-10-23) and round 3b
`emojet-v2-round3` (from round 2): slang probes **32 → 46/72** (top 3: 23 → 40), Bengali 18 → 28/44, Hindi 8 → 10/19,
English 6 → 8/9, lookalikes 5 → 6/8, chat cases 22 → 24/40 top-1 and 30 → 33/40 top-3, emoji cases 24/24. Phone file
(Q4_K_M): 46/72 plain, 51/72 with the app's ranking. **Not shipped**: real held-out messages got worse (Bengali top-1
25.4 → 22.9%, top-6 62.3 → 59.7%; Hindi top-1 26.9 → 24.2%, top-6 55.4 → 53.5%), the owner's own "Haga pacche"
(phone-capitalised) still missed, and toilet emojis leaked into ordinary Banglish ("khete jacchi" → …🚽).
Causes: (1) the starting-model evaluation ran inside the training timer, so the run planned 29k steps (0.38 epoch) and
stopped after 6.7 of 10 h; (2) three meaning emojis at 0.4 each outweighed one typed emoji (55% of the target);
(3) Bengali ×2 / Hindi ×1 and replay 40-80% cut real-message exposure; (4) no capitalised copies (phones capitalise).

Round 3b (`emojet-v2-round3b`, continues from round 3): timer fixed; typed emojis keep >= 60% of the target (labeller 20%,
meanings 20%); Bengali ×3, Hindi ×2, real Hindi/Bengali not subsampled (English 50%); spelling variants also on repeated
labelled copies, 40% of variants capitalised; new metric `slang_capitalised_hit@6`; the comparison table has a round-2
column (the shipped model, the bar to beat). Same data kernels; start model `emojet-v2-round3`.
```bash
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round3 && python3 scripts/kaggle.py push training/kaggle/emojet_v2_round3.ipynb emojet-v2-round3b --gpu --internet \
    --source=subhadeep3902/emojet-v2-round3 --source=subhadeep3902/emojet-data --source=subhadeep3902/emojet-tones --source=subhadeep3902/emojet-slang-3 \
    --source=subhadeep3902/emojet-annotate --source=subhadeep3902/emojet-annotate-2 --source=subhadeep3902/emojet-annotate-3
```

### Round 3b result (2026-10-23) and the model soup
`emojet-v2-round3b` (from round 3, 9.2 h, 41k steps): slang probes 49/72 (top 3: 44/72), capitalised 44/72, Hindi probes
13/19, chat cases 25/40 top-1, 32/40 top-3. Real held-out still below round 2 (Bengali top-1 23.2% vs 25.4%, top-6 58.7%
vs 62.3%; Hindi top-1 24.3% vs 26.9%): teaching chat-word meanings costs a little on reviews/comments/tweets, which is
what the real held-out set mostly is. Not shipped as is.

Next: `training/build_soup.py` blends round 2 and round 3b weights (WiSE-FT) at a = 0, 0.3 … 0.7, 1, evaluates each blend
with the training evaluation, and saves the best blend with no regression vs round 2 (real held-out within 0.5 point,
chat/emoji cases, tones, style, sarcasm, lookalikes) and the most slang skill. ~40 min GPU.
```bash
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round3 && python3 training/build_soup.py $(git rev-parse --short HEAD) && \
python3 scripts/kaggle.py push training/kaggle/emojet_soup.ipynb emojet-v2-soup --gpu --internet \
    --source=subhadeep3902/emojet-v2-round2 --source=subhadeep3902/emojet-v2-round3b --source=subhadeep3902/emojet-data
```

### Soup result (2026-10-24): shipped as Emojet-v3, and what is still broken
Second soup run (gate: paired McNemar per language, a drop blocks only if >2 points or p<0.05, plus a required slang win).
Blend **alpha 0.5** passed and is shipped as `models/Emojet-v3-Q4_K_M.gguf` (sha256 b255d653…). 0.6-1.0 all failed on
statistically real drops (0.7: en hit@6 p=0.010, bn hit@6 p=0.044; 1.0: bn hit@6 −3.5 pt p<0.001).

Phone file (Q4_K_M) vs the previous phone model: slang probes 29 → **41/72** (bn 16 → 24, hi 7 → 9, en 6 → 8),
hit@3 24 → 34, lookalikes 5 → **6/8**, emoji cases 22 → 23/24, chat top-3 29 → 30/40, chat top-1 22 → 21/40 (one case,
quantization noise). Real held-out: no significant change.

**Still failing — the owner's own phrases.** Short Banglish/Hinglish of the exact form he types:
`Haga pacche` → 😂😅😮😢🤣🎉 (no 💩🚽 in any model), `Brishti porche baire` → 😂😅🤮🤬🖕, `Khide peyeche` → ❤️😊😅,
`tatti aa rahi hai` → 😂😅🤣. Round 3b alone *does* get `Ghum pacche` → 💤😴🛌 and `Khide peyeche` → 🍛, but it is the
variant that regresses real held-out, and the 0.5 blend dilutes those wins away.
Two separate causes:
1. **Too little signal for the crude words.** "haga" has 112 real messages in 2.9 M, and in most of them the writer typed
   😂 (laughing at someone), plus ~48 synthetic lines at weight 0.5. "ghum" has 700 and works. Fix: generate far more
   synthetic lines for the ~20 words the owner actually uses, weight them like real data, and keep the topic emoji
   (💩🚽) as a hard target for them.
2. **Tone is mis-read on short romanized Bengali.** "Brishti porche baire" scores rude/angry 0.93, which is why angry
   emojis rank at all. The app-side guard (`OFFENSIVE` in `Ranker.kt` / `rank_sim.py`: 🖕🤬🤮🤢🍆🍑👿 suppressed unless
   the message scores rude/aggressive/angry/derisive ≥ 0.5) removes them when the tone is read correctly
   (`tatti aa rahi hai` is clean now) but cannot help when the tone itself is wrong. Fix: tone training data for short
   romanized Bengali (the tone sets are English-heavy).
