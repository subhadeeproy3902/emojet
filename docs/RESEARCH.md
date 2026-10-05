# Research notes

## 1. Jev vs Laya

[emoji-jev](https://github.com/colinmcdermott/emoji-jev) is the inspiration: real-time emoji autocomplete that asks several typed questions about your draft in one model call (emoji choice from 64/128/254, emotion, mood/urgency/energy scores, sarcasm/joke/reply-wanted). It runs on **Jev**, a hosted model from TypeSafe AI, billed per call (~$0.0002 per keystroke). Not usable at $0.

[Laya](https://huggingface.co/convaiinnovations/laya) (convaiinnovations, Apache-2.0) is the open equivalent: a **non-autoregressive decision model** (ModernBERT encoder + decision head, trained with RL against proper scoring rules) that answers typed questions (`choice` / `score` / `noul` yes-no) in one forward pass per question. No billing, runs offline.

| Checkpoint | Base | Params | GGUF available |
|---|---|---|---|
| `laya` | ModernBERT-large | 421M | **yes** (`ggml-org/Laya-GGUF`) |
| `laya-multilingual` | mmBERT-base | 322M | no (safetensors only) |
| `laya-typed-decisions` | ModernBERT-large | 421M | no (safetensors only) |

## 2. Runtime: llama.cpp

- Laya support (+ julia-1, lev, openjev, kev) was added to llama.cpp in [PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818), merged **2026-10-02**. It's newer than the Arch `llama-cpp` package, so llama.cpp is built from source.
- API: `POST /v1/systemone` on `llama-server`, compatible with the TypeSafe (Jev) API: `{state, questions: {id: {type, instructions, criteria}}}` → `{answers: {id: {choice, probabilities, confidence}}}`.
- Laya: up to **255 options** per choice question; each question is a separate forward pass (`n_variants = 1`); options + question share a **192-token budget** (`modern-bert.decision.max_head_tokens`), longer ones are cut evenly "the same way the model was trained".
- The prompt is `[cls] question [sep] ([mask] option)* [sep] state [sep]`; it's an encoder, so no prefix/KV caching between requests — cost is linear in prompt length.
- The decision logic lives in `tools/server/server-decision.cpp` (~800 lines), not in libllama → see ARCHITECTURE for how each client embeds it.
- Other Laya artifacts: `desplega/laya-onnx` (fp32 only, 1.3–1.7 GB, Node/Bun) — not used. `laya.tools/platform/gguf` lists ~27 community projects (servers, a desktop app, an npm package); none target Android or browsers.

## 3. Phase 0 experiments (2026-10-04, laptop: i5-12450H, CPU only)

### 3.1 Asking for emojis directly — fails

64 emojis as `choice` options (Jev-style), plus emotion + sarcasm questions:

- Picks were close to random (😂 for "thank you so much", 💜 for "huge spider help", ✌️ for "happy birthday").
- Cause: ❤️ alone is 6 tokens, 😂 is 3; with 64 options in a 192-token budget each option is cut to ~3 tokens → the model compares broken byte fragments, and descriptions are cut away entirely.
- Also slow: ~1.4 s for three questions (~360 tokens each).
- The **emotion** question (8 word labels) was mostly right in the same run → Laya understands the text; the option format was the problem.

### 3.2 Mood labels → emoji rows — works

25 word labels in one `choice` question (`shared/moods.json`), app maps top moods to emoji rows. 23 hand-labelled samples (incl. Hinglish and 3 reply cases), "correct" = a mood a human would accept.

| Variant (Q8_0, 8 threads) | Top-1 | Top-3 | Median latency | Prompt |
|---|---|---|---|---|
| labels + 2-word descriptions | 17/23 | 20/23 | 390–600 ms | ~155 tok |
| labels only | 17/23 | 21/23 | 334–390 ms | ~80 tok |

Descriptions add tokens without helping → off by default.

### 3.3 Model file size (labels only, 8 threads)

| File | Size | Top-1 | Top-3 | Median |
|---|---|---|---|---|
| Q8_0 (official) | 449 MB | 17 | 21 | 334 ms |
| Q4_K_M (made locally) | 287 MB | 17 | 20 | 250 ms |
| **Q4_0** (made locally) | **252 MB** | **18** | **22** | **196 ms** |
| BF16 (official) | 844 MB | 16 | 21 | 267 ms |

Differences of ±1 sample are noise at n=23, but Q4_0 is clearly not worse and is the smallest and fastest (x86 has fast repacked Q4_0 kernels; ARM does too). **Current pick: Q4_0.**

### 3.4 Threads and memory (Q4_0, lean server: ctx/batch 512, 1 slot, no prompt cache)

- 8 threads: 188 ms · 4 threads: 206 ms · 2 threads: 335 ms → 4 threads is the sweet spot.
- Memory: ~265 MB real (anonymous: weights repacked for fast CPU kernels + 25 MB compute buffer) + ~260 MB of the model file in the page cache, which the OS can drop.
- Cost ≈ 2.5 ms per prompt token on this CPU (4-option question = 130 ms, 25-option question = ~190 ms).

### 3.5 Reply awareness — confirmed useful

| | reply-same ("same here") | reply-where ("where exactly?") | reply-yes ("yes!!" to a concert invite) |
|---|---|---|---|
| with replied-to message | agree ✓ | travel ✓ (📍) | celebration ✓ |
| draft only | neutral | neutral | neutral ✗ |

### 3.6 Known misses (tuning backlog)

- angry delivery complaint → `sad`; "another monday meeting that could have been an email" → `neutral` (sarcasm); "you looked really cute today" → `proud`; "ok see you at 5" → `goodnight`; "sent that text to my boss" → `sorry` (awkward was wanted).
- Probabilities are very peaky (often 1.00), so blending the top 3 moods rarely mixes rows. Try flattening on the client (e.g. `p^0.3` before blending) so the 2nd mood contributes.
- 23 samples is too few to tune on without overfitting → needs the owner's ~50 real-style messages (MANUAL-WORK §B).

## 4. Phone measurement (2026-10-05) — Phase 1 answered

Phone: **REDMI Note 17 Pro 5G** (HyperOS 3.0, Android 16), Qualcomm SM6435 — 4× Cortex-A78 + 4× Cortex-A55, dot-product instructions (no i8mm), Adreno GPU, 8 GB RAM.

Build: `llama-server` cross-compiled with NDK r30, `arm64-v8a`, **KleidiAI on** (runtime-dispatched Arm kernels for Q4_0/Q8_0 → one portable build for any arm64 phone), stripped to 13 MB. Run from `/data/local/tmp` over adb, laptop sends the same 23 samples (`bash scripts/phone-bench.sh`).

| Threads | Median per suggestion |
|---|---|
| 2 | 559 ms |
| **4** | **306–356 ms** |
| 6 | 455 ms |
| 8 | 387 ms |

- Same accuracy as the laptop (18/23 first mood, 22/23 top 3) — the phone build gives identical answers.
- Real memory ~230–254 MB (plus the model file in the page cache, which Android can drop).
- 4 threads = the 4 fast cores; adding the slow cores makes it slower. The app will use the number of fast cores (fallback 4).
- Target was ≤ 600 ms → met with margin on CPU alone. GPU backends (OpenCL/Vulkan on Adreno) are optional future work, not needed.
- Laya-Emoji's prompt is just the message (~25–40 tokens vs ~80 now), so it should be roughly 2× faster on the same phone.

## 5. Tone of voice (2026-10-04)

### 5.1 Is there one official list?

No. Published lists range from 25 to 100+ tones and are mostly synonyms of each other. EmoJet's taxonomy ([`shared/tones.json`](../shared/tones.json)) merges and de-duplicates:

| Source | What it contributed |
|---|---|
| Owner's brief | 4 communication styles (assertive, passive, aggressive, passive-aggressive) + 16 tones in 4 families |
| [Nielsen Norman Group — Tone-of-Voice Words](https://www.nngroup.com/articles/tone-voice-words/) | 4 tone dimensions (funny↔serious, formal↔casual, respectful↔irreverent, enthusiastic↔matter-of-fact) + 37 tone words |
| [PapersOwl — Types of Tone](https://papersowl.com/blog/types-of-tone) | ~85 tones grouped positive / neutral / negative |
| [Grammarly tone detector](https://www.grammarly.com/tone) ([TechCrunch](https://techcrunch.com/2019/09/24/GRAMMARLY-GETS-A-TONE-DETECTOR-TO-KEEP-YOU-OUT-OF-EMAIL-TROUBLE/), [VentureBeat](https://venturebeat.com/ai/grammarly-uses-ai-to-detect-the-tone-and-tenor-of-your-writing)) | 40+ detected tones, each shown with an emoji — precedent for tone → emoji |
| [The Write Practice — 42 tones](https://thewritepractice.com/tone-in-writing/), [selfpublishing.com — 25 tones](https://selfpublishing.com/writing-tones/) | cross-check |

Result: **4 styles + 46 canonical tones** (each with aliases, e.g. condescending = patronizing / arrogant / superior / smug) in 5 families, + the 4 NN/G dimensions. Added beyond the owner's list because they matter for chat emojis: flirty, nostalgic, humorous, loving, sad, anxious, surprised, disappointed, confused, embarrassed, urgent, rude, bitter, demanding.

### 5.2 Zero-shot test: untrained Laya (Q4_0, `scripts/test-tones.py`, 25 labelled messages)

- Tone: **14/25** first pick, **18/25** top 3. Style: **6/13** (biased to "passive").
- Correct: apologetic, empathetic, anxious, surprised, encouraging, playful, appreciative, confused/tentative, formal/neutral.
- Wrong: sarcasm ("another meeting, my favourite thing ever" → joyful), passive-aggressive ("since you always know best" → neutral), condescending ("you wouldn't understand" → analytical), accusatory ("you always mess this up" → embarrassed), defensive, cynical, nostalgic.
- ~170 prompt tokens per question (46 options) → ~1.15 s for style + tone. Too slow for runtime as separate questions; fine for offline teacher labelling.
- Style descriptions didn't help (5/13).

Conclusion: train tone as extra outputs of Laya-Emoji (same pass, no extra cost), with real data for the implicit tones (TRAINING.md).

### 5.3 Free tone datasets found on Hugging Face (licence-checked)

| Dataset | Tone signal | Licence | Use |
|---|---|---|---|
| google-research-datasets/go_emotions | 27 emotions (maps to ~20 tones) | Apache-2.0 | yes |
| CreativeLang/SARC_Sarcasm | sarcasm (Reddit "/s") | CC-BY-2.0 | yes |
| Cleanlab/stanford-politeness | politeness | MIT (orig. CC-BY-4.0) | yes |
| osyvokon/pavlick-formality-scores | formality | CC-BY-3.0 | yes |
| tasksource/jigsaw_toxicity | insult / threat / toxicity | Apache-2.0 | yes |
| ToneStyle/TST2K | tone style | CC-BY-NC-ND-4.0 | **no** (non-commercial, no derivatives) |
| EmpatheticDialogues | empathy | CC-BY-NC-4.0 | **no** |
| raquiba/Sarcasm_News_Headline, marcbishara/sarcasm-on-reddit, ToneStyle/TST100K | various | no licence stated | **no** |

No permissive dataset exists for passive-aggressive, condescending, defensive or the 4 communication styles → hand-written examples (owner + Claude) and careful teacher labels.

## 6. Laya internals (from its reference code, 2026-10-05)

- `rl_common.DecisionModel` = ModernBERT-large encoder (28 layers, hidden 1024, fine-tuned by Laya's authors) + 2 plain pre-norm Transformer layers + a per-option scorer on each option's `[MASK]` marker; `act_head` is an extra "escalate" output llama.cpp doesn't expose. Trained 1 epoch, 7,313 updates, 1.96 h on one GPU.
- Temperatures (`rl_agent_config.json`): per question type plus per option-count bucket — **`choice:11+` = 0.1006**, which is why 25-mood / 46-tone answers look like "1.00". To recover softer probabilities, raise them to the power ≈ 0.1 (`p^0.1006`, renormalised) before blending.
- llama.cpp's `/v1/systemone` reproduces this code exactly (§ TRAINING step 1: 20/20 answers).
