# Contributing to Emojet

Emojet is the model behind the [EmoJet app](https://github.com/subhadeeproy3902/emojet-ai). Anyone can retrain it,
improve it, or point it at a new language — everything runs on free hardware (a Kaggle T4), and every run is a
notebook this repo generates for you.

There are four useful ways to help, roughly in order of how much time they take.

## 1. Report a bad suggestion (5 minutes, genuinely valuable)

Open a [Bad emoji suggestion](../../issues/new?template=bad_suggestion.yml) issue with the message you typed and the
emojis a real person would send. These become test cases in `training/data/eval_sets.json`, which is how a complaint
turns into something that cannot silently regress again.

## 2. Add test cases (no GPU needed)

`training/data/eval_sets.json` holds every test the model is judged on:

| Set | What it checks |
|---|---|
| `chat_cases` | 40 WhatsApp-style messages — the emoji must be in the top 3 |
| `slang_probes` | 72 slang / everyday Banglish-Hinglish messages — hit@6 |
| `slang_negatives` | lookalike words that must **not** get toilet or swear emojis (`Bhagavad Gita`, `chhod do`) |
| `emoji_cases`, `tone_cases` | concepts and tone of voice |

Add rows, open a PR. Keep them realistic: things you would actually send, not dictionary sentences.

## 3. Add words to the chat lexicon (no GPU needed)

`training/data/chat_lexicon.json` teaches the labeller and the training run what everyday chat words mean. One entry:

```json
{"lang": "bn", "kind": "topic",
 "roman": ["ghum", "ghumabo", "ghum pacche"], "native": ["ঘুম", "ঘুম পাচ্ছে"],
 "gloss": "sleep / sleepy / going to sleep", "emojis": "😴💤🛌"}
```

- `kind`: `topic` (the word names a thing or state — its emojis always apply), `reaction` (a whole-message reaction,
  only applies to short messages), `swear` or `address` (meaning only, no emojis).
- Spellings are matched as **whole words** with spelling folding, so you do not need every variant —
  `pacche`/`pachhe`/`pache` all match one entry. Run `python3 -c "..."` against `training/chatlex.py` to check.
- Ambiguity is the enemy: before adding a short word, check it is not an ordinary word in the other language
  (`baje` is "o'clock", `cha` is "want", `mut` folds into "mot"). CI fails if an entry matches a known lookalike.

## 4. Retrain the model (free, ~10 h on a Kaggle T4)

Read [`docs/TRAINING.md`](docs/TRAINING.md) — it has the full pipeline, every round that was run, and what went wrong
in each. The short version:

```bash
python3 training/build_notebook.py $(git rev-parse --short HEAD) --round3   # builds the notebook
python3 scripts/kaggle.py push training/kaggle/emojet_v2_round3.ipynb my-run --gpu --internet --source=...
```

Every run evaluates the starting model first, again halfway, and prints a table at the end that marks anything
that got **worse** than the shipped model. Please include that table in your PR.

Three rules learned the hard way, all of them from real mistakes in this repo's history:

1. **Match whole words, never substrings.** Round 2 matched `haga` inside `bhagao` and half its slang data was junk.
2. **A 1-point difference on 2,000 held-out messages is noise.** Use the paired test in `training/build_soup.py`
   before claiming an improvement.
3. **Never trade real-message accuracy for a test-set win** without saying so out loud in the PR.

## Setting up

```bash
git clone https://github.com/subhadeeproy3902/emojet && cd emojet
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers datasets regex   # only needed for local training work
```

Running things locally against a model file:

```bash
MODEL=models/Emojet-v1-Q4_K_M.gguf bash scripts/dev-server.sh &   # needs llama.cpp, see scripts/build-engine.sh
python3 scripts/rank_sim.py                                       # the full test suite, the way the app ranks
```

## House rules

- **Never commit** `.env`, model weights, `*.gguf`, `*.safetensors` or anything under `data/`.
- Any dataset you add must allow this use, and its licence goes next to the loader in
  `training/real_data.py` or `training/tone_data.py`.
- Code is Apache-2.0; the **weights** are CC BY-NC-SA 4.0 because some training data is non-commercial.
  Contributions are taken under those same terms.
- Be accurate about numbers. "It feels better" is a reason to measure, not a result.
