# Emojet

**Emojet** reads a whole chat message — and the message it replies to — and, in **one forward pass**, scores **every emoji in Unicode 18.0** plus the **tone of voice** it's written in. It runs fully offline on phones and laptops.

It is a fine-tune of [Laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0), the open decision model built on ModernBERT-large. The app that uses it is [EmoJet AI](https://github.com/subhadeeproy3902/emojet-ai) — emoji suggestions over any Android app.

| Version | Status |
|---|---|
| **Emojet-v1** | trained 2026-10-05: 3 epochs + 1 epoch with chat-object data (this README) |
| Emojet-v1-mini | planned: half the layers, distilled from v1, ~half the size |

## What it outputs

One pass → **1,977 scores**:

| Group | Count | Activation | Examples |
|---|---|---|---|
| emojis | 1,923 | softmax over the group | every distinct emoji of Unicode 18.0 (skin tones applied by the app) |
| tones | 46 | sigmoid each | sarcastic, empathetic, condescending, playful, apologetic, … |
| communication styles | 4 | softmax | assertive, passive, aggressive, passive-aggressive |
| tone dimensions | 4 | sigmoid each | funny↔serious, formal↔casual, respectful↔irreverent, enthusiastic↔matter-of-fact |

Label order: [`labels/emoji-catalog.json`](labels/emoji-catalog.json) (emojis), then [`labels/tones.json`](labels/tones.json) (tones, styles), then the 4 dimensions — also stored in `emojet.json` next to the weights.

**Input text** (build it exactly like this):

```
replying to: <the message you're answering>      ← only when replying
my message: <the draft>
```

## Model details (Emojet-v1)

| | |
|---|---|
| Architecture | `ModernBertForSequenceClassification`, mean pooling |
| Parameters | ≈ 398 M (encoder ≈ 396 M from Laya, new output layer ≈ 2 M) |
| Layers / hidden / heads | 28 / 1,024 / 16, GeGLU MLP (2,624) |
| Vocabulary | 50,368 (ModernBERT BPE) |
| Context used | 64 tokens (message + reply); the encoder supports 8,192 |
| Base | `convaiinnovations/laya` @ `7b928d82` (its encoder; Laya's decision head is not used) |
| Weights | `safetensors` fp16 (~800 MB) · GGUF F16 (~797 MB) · **GGUF Q4_K_M (273 MB, phones)** · GGUF Q8_0 (424 MB) |
| Speed | ~0.1–0.2 s on a laptop CPU; ~0.2–0.4 s on a mid-range phone CPU (4 big cores) |

## Results (held-out data, never trained on)

| Test | Untrained Laya (mood question) | **Emojet-v1** |
|---|---|---|
| Emoji hit — an acceptable emoji in the top 6 (24 chat messages) | — | **23 / 24** |
| Tone of voice in the top 3 (25 messages) | 18 / 25 | **22 / 25** |
| Communication style (13 messages) | 6 / 13 | **11 / 13** |
| Concepts → emoji, emojinize test (1,500 sentences) | — | **99.3 %** |
| Feelings → emoji, GoEmotions test | — | **86.8 %** |
| Hinglish sarcasm (2,344 headlines) | — | **82.0 %** |
| Distinct emojis used across the tests | ≤ ~150 (mood rows) | **542** |

Chat about things now works ("I'm watching television" → 📺, "happy diwali" → 🪔, "pizza tonight?" → 🍕, "fever" → 😷🤒, "India won" → 🇮🇳🏏) after a 4th epoch with ~4,000 chat-object examples. Known weaknesses (v2): some animals ("puppy" → 👶), 🔥🙌🤩 still frequent, Romanized Bengali/Hindi still thin in the data.

Quantization (llama.cpp): **Q4_K_M** (273 MB) keeps the scores within ~1.1 of F16 and the same test results; Q4_0 drifts too much for this classifier; Q8_0 (424 MB) is near-exact.

## Run it locally

**Python (transformers ≥ 4.48):**

```python
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification
repo = "subhadeeproy3902/emojet"             # Hugging Face, once published
tok = AutoTokenizer.from_pretrained(repo)
model = ModernBertForSequenceClassification.from_pretrained(repo).eval()
labels = model.config.id2label                # 0-1922 emojis, then "tone:…", "style:…", "dim:…"
text = "replying to: do you want to come to the concert?\nmy message: yes!!"
with torch.no_grad():
    enc = tok(text, return_tensors="pt")
    s = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0]
print([labels[i] for i in s[:1923].topk(6).indices.tolist()])
tones = torch.sigmoid(s[1923:1969])
print([labels[1923 + i] for i in tones.topk(3).indices.tolist()])
```

**llama.cpp** (CPU, phones, servers): needs [`engine/llama.cpp-patches/0001-…`](engine/llama.cpp-patches) until it's upstream — stock `llama-server` returns only the first 1,024 of the 1,977 scores.

```bash
bash scripts/build-engine.sh host                       # pinned llama.cpp + patch
MODEL=models/Emojet-v1-Q4_0.gguf bash scripts/dev-server.sh &
curl -s localhost:8080/embedding -d '{"content":"my message: landed in goa, beach time","embd_normalize":-1}'
python3 scripts/test-emojet.py                           # scores the test sets
```

## Training

- Data (≈ 305,500 examples, all licences allow this use): [emojinize-multilingual](https://huggingface.co/datasets/yagizgencer/emojinize-multilingual) (CC-BY-4.0), [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Apache-2.0), [Hinglish/English sarcasm headlines](https://huggingface.co/datasets/BM2002/hinglish-sarcasm-dataset) (CC-BY-4.0), [SemEval-2018 irony tweets via Natural Instructions](https://huggingface.co/datasets/Lots-of-LoRAs/task386_semeval_2018_task3_irony_detection) (Apache-2.0), [Stanford Politeness](https://huggingface.co/datasets/Cleanlab/stanford-politeness) (MIT), [Pavlick formality](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) (CC-BY-3.0), [Jigsaw toxicity](https://huggingface.co/datasets/tasksource/jigsaw_toxicity) (Apache-2.0), Unicode CLDR names/keywords (Unicode-3.0), 129 hand-written English/Hinglish/Banglish seeds ([`training/data/seeds.jsonl`](training/data/seeds.jsonl)), synthetic reply pairs.
- Loss: soft-target cross-entropy over emojis, masked BCE for tones/dimensions, cross-entropy for style; each example only teaches the outputs its source knows about.
- 3 epochs, batch 64, AdamW (3e-5 encoder / 5e-4 head), fp16, one free Kaggle T4, 126 min; then 1 epoch at 1.5e-5 / 1e-4 with the chat-object examples added (42 min).
- Everything is reproducible: `python3 training/build_notebook.py <commit>` → push to Kaggle with `scripts/kaggle.py` (see [`docs/TRAINING.md`](docs/TRAINING.md)).

## Licence

Apache-2.0 (see [`LICENSE`](LICENSE)). Emojet-v1 is a modified version of Laya (Apache-2.0, © its authors): Laya's encoder was fine-tuned and a new classification layer was added. Training data credits and licences: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
