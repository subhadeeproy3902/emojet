#!/usr/bin/env python3
"""Build training/kaggle/laya_emoji_train.ipynb (the Kaggle training notebook) for a given repo commit.

  python3 training/build_notebook.py <commit> [--smoke]   # commit must be pushed; --smoke = quick 4,000-example test run
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "main"
SMOKE = "--smoke" in sys.argv
MINI = "--mini" in sys.argv     # distil Emojet-v1 into a half-depth student
RESUME = "--resume" in sys.argv # continue training Emojet-v1 for one epoch (adds chat-object data)
V2 = "--v2" in sys.argv         # Emojet-v2 (pre): from final v1, topic-in-chat data, balanced emoji loss
REAL = "--real" in sys.argv     # Emojet-v2: from v2-pre + real messages (EN/HI/BN) from the emojet-data kernel
ROUND2 = "--round2" in sys.argv # Emojet-v2 round 2: continue from emojet-v2-real, + slang data, more Banglish/Hinglish weight
ROUND3 = "--round3" in sys.argv # Emojet-v2 round 3: continue from round 2; clean slang + chat-word meanings + synthetic chat
LLAMA_CPP_COMMIT = "2e7c58c5477478c8cf6e199cfaa5dcd5a4319c81"
# round 2 = the model on the phone: every new round must beat it (round 3 measured it as its starting point)
_r3 = ROOT / "data/kaggle/emojet-v2-round3/emojet-v2/emojet.json"
REFERENCE = json.loads(_r3.read_text())["baseline"] if _r3.exists() else {}

import base64 as _b64, gzip as _gz
_FILES = ["labels/emoji-catalog.json", "labels/tones.json", "training/data/goemotions_map.json",
          "training/data/seeds.jsonl", "training/data/eval_sets.json", "training/chatlex.py", "training/data/chat_lexicon.json"]
EMBED = _b64.b64encode(_gz.compress(json.dumps({f: (ROOT / f).read_text(encoding="utf-8") for f in _FILES}).encode())).decode()

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip()})


md(f"""
# Emojet — training **{"Emojet-v1-mini (distilled)" if MINI else ("Emojet-v2 (from v1, topic-in-chat data, balanced loss)" if V2 else ("Emojet-v1, continued with chat-object data" if RESUME else "Emojet-v1"))}**

Fine-tunes [Laya](https://huggingface.co/convaiinnovations/laya)'s encoder (ModernBERT-large, 28 layers, Apache-2.0) into one model that reads a **whole message** (plus the message being replied to) and outputs, in a single pass:

| Group | Outputs |
|---|---|
| emojis | 1,923 — every distinct emoji of Unicode 18.0 (skin tones applied by the app) |
| tones | 46 tones of voice (sarcastic, empathetic, condescending, …) |
| styles | 4 communication styles (assertive, passive, aggressive, passive-aggressive) |
| dimensions | 4 sliders (funny↔serious, formal↔casual, respectful↔irreverent, enthusiastic↔matter-of-fact) |

Repo: https://github.com/subhadeeproy3902/emojet (data files from commit `{COMMIT[:10]}`). Plan: `docs/TRAINING.md`.

**Baseline to beat (untrained Laya, mood labels, laptop):** right mood first try 18/23, top-3 22/23; tone top-3 18/25; style 6/13; Banglish "Bhai ami khete jacchi, pore kotha bolchi" → 👋😊🤗🙌✌️😄 (misses the food).
""")

md("## 1. Settings")
code(f"""
REPO, COMMIT = "subhadeeproy3902/emojet", "{COMMIT}"
EMBEDDED_B64 = "{EMBED}"
LAYA_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"      # convaiinnovations/laya, same as the GGUF we run on phones
LLAMA_CPP_COMMIT = "{LLAMA_CPP_COMMIT}"
SMOKE = {SMOKE}                                            # True = quick end-to-end test on 4,000 examples
MINI = {MINI}                                              # True = distil Emojet-v1 into Emojet-v1-mini
RESUME = {RESUME}                                          # True = continue Emojet-v1 for one epoch
V2 = {V2}                                                  # True = train Emojet-v2 (pre) from the final v1
REAL = {REAL}                                              # True = Emojet-v2 on real messages, from v2-pre
ROUND2 = {ROUND2}                                          # True = continue Emojet-v2 (round 2): slang + more Banglish/Hinglish
ROUND3 = {ROUND3}                                          # True = round 3(b): clean slang, word meanings, synthetic chat
REFERENCE = {json.dumps(REFERENCE)}                         # round-2 metrics (the shipped model)
if ROUND2 or ROUND3: REAL = True
LANG_REPEAT = {{"bn": 3, "hi": 2, "en": 1}} if (ROUND2 or ROUND3) else {{"bn": 2, "hi": 1, "en": 1}}   # times each real message is seen
# round 3: data seen in earlier rounds is replayed at a fraction (no forgetting), so most of the time goes to new data
REPLAY = {{"emojinize": 0.5, "emojinize-span": 0.5, "goemotions": 0.6, "chat-objects": 0.5, "topic-chat": 0.5, "toxicity": 0.5,
          "formality": 0.5, "politeness": 0.6, "sarcasm-headlines": 0.6, "irony-tweets": 0.6, "real-en": 0.5,
          "real-long": 0.5}} if ROUND3 else {{}}
if REAL: V2 = True                                         # keeps the v2 data (topic-in-chat) and loss
NAME = "emojet-v1-mini" if MINI else ("emojet-v2" if V2 else "emojet-v1")
BALANCE_TAU = 0.5 if REAL else (0.6 if V2 else 0.0)        # balanced softmax: frequent emojis stop crowding out specific ones
KEEP = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25]  # teacher layers kept (every 3rd = global attention, pattern preserved)
KD_T, KD_W, SUP_W = 2.0, 1.0, 0.5
MAX_LEN, BATCH, EPOCHS = (256 if REAL else 64), 64, (1 if (SMOKE or RESUME) else (2 if (MINI or V2) else 3))   # the time budget cuts it
LR_ENCODER, LR_HEAD, WARMUP = 3e-5, 5e-4, 0.05
if V2:
    LR_ENCODER, LR_HEAD, WARMUP = 2e-5, 1.5e-4, 0.03
if ROUND2 or ROUND3:                                       # refining an already trained model: gentler steps
    LR_ENCODER, LR_HEAD, WARMUP = 1e-5, 8e-5, 0.02
if RESUME:
    LR_ENCODER, LR_HEAD, WARMUP = 1.5e-5, 1e-4, 0.03
TONE_W, STYLE_W, DIM_W = 1.0, 0.5, 0.5
TIME_BUDGET_H = 10.0 if REAL else 7.0                      # Kaggle stops a session at 12 h; the LR schedule is fitted to this
SEED = 42
# The exact text the model sees. Apps must build the same string (shared/laya-emoji.json).
def fmt(draft, reply=None):
    return (f"replying to: {{reply.strip()}}\\n" if reply else "") + f"my message: {{draft.strip()}}"

import json, math, os, random, re, subprocess, sys, time, urllib.request
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation (set before torch)
import numpy as np, torch, torch.nn.functional as F
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
import transformers, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
print("GPU:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
T_START = time.time()
""")

md("## 2. Label space (from the repo's label files, embedded below)")
code("""
# Data files are embedded in the notebook (works while the repo is private).
import base64, gzip
EMBEDDED = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_B64)).decode("utf-8"))
def get(path):
    return EMBEDDED[path]

catalog  = json.loads(get("labels/emoji-catalog.json"))["emojis"]
tones_cfg = json.loads(get("labels/tones.json"))
ge_map   = json.loads(get("training/data/goemotions_map.json"))
seeds    = [json.loads(l) for l in get("training/data/seeds.jsonl").splitlines() if l.strip()]
evals    = json.loads(get("training/data/eval_sets.json"))

EMOJIS = [e["e"] for e in catalog]
TONES  = list(tones_cfg["tones"])
STYLES = list(tones_cfg["styles"]["options"])
DIMS   = [k for k in tones_cfg["dimensions"] if k != "note"]
LABELS = EMOJIS + [f"tone:{t}" for t in TONES] + [f"style:{s}" for s in STYLES] + [f"dim:{d}" for d in DIMS]
NE, NT, NS, ND = len(EMOJIS), len(TONES), len(STYLES), len(DIMS)
OFF_T, OFF_S, OFF_D = NE, NE + NT, NE + NT + NS
N_LABELS = len(LABELS)
TONE_IDX = {t: i for i, t in enumerate(TONES)}
STYLE_IDX = {s: i for i, s in enumerate(STYLES)}
print(f"{NE} emojis + {NT} tones + {NS} styles + {ND} dims = {N_LABELS} outputs; {len(seeds)} seeds")

import regex
SKIN = {chr(c) for c in range(0x1F3FB, 0x1F400)}
def norm(e): return "".join(ch for ch in e if ch not in SKIN and ch != "\\ufe0f")
E_INDEX = {}
for i, e in enumerate(EMOJIS):
    E_INDEX.setdefault(norm(e), i)

def emoji_ids(s):
    out = []
    for g in regex.findall(r"\\X", s or ""):
        i = E_INDEX.get(norm(g))
        if i is not None and i not in out:
            out.append(i)
    return out

def strip_emojis(s):
    return re.sub(r"\\s+", " ", "".join(g for g in regex.findall(r"\\X", s or "")
                                       if not (norm(g) in E_INDEX and any(ord(c) > 0x2000 for c in g)))).strip()

print(emoji_ids("party 🎉🥳 👍🏽 ❤️"), strip_emojis("party 🎉🥳 tonight 👍🏽"))

# chat lexicon (training/chatlex.py): what Banglish/Hinglish chat words and slang mean, and their emojis
_ns = {}
exec(get("training/chatlex.py"), _ns)
LX = _ns["Lexicon"](json.loads(get("training/data/chat_lexicon.json"))["entries"])
def lex_ids(text, lang):
    out = []
    for s in LX.emojis(text, lang):
        out += [i for i in emoji_ids(s) if i not in out]
    return out[:4]
def mix(gold=(), silver=(), topic=()):
    # One soft target. What the writer typed keeps >= 60% (labeller's extra emojis 20%, the words' meanings 20%):
    # round 3 let 3 meaning emojis outweigh 1 typed emoji and lost 2-3 points on real messages.
    # Without typed emojis: labeller 60%, meanings 40%.
    gold = list(dict.fromkeys(gold or ()))[:6]
    silver = [i for i in dict.fromkeys(silver or ()) if i not in gold][:4]
    topic = [i for i in dict.fromkeys(topic or ()) if i not in gold and i not in silver][:4]
    if gold:
        parts = [(silver, 0.2), (topic, 0.2)]
        parts = [(g, sh) for g, sh in parts if g]
        parts = [(gold, 1.0 - sum(sh for _, sh in parts))] + parts
    else:
        parts = [(g, sh) for g, sh in ((silver, 0.6), (topic, 0.4)) if g]
        tot = sum(sh for _, sh in parts)
        parts = [(g, sh / tot) for g, sh in parts]
    if not parts:
        return None, None
    ids, p = [], []
    for g, sh in parts:
        for i in g:
            ids.append(i); p.append(sh / len(g))
    return ids, p
VARY = [("chh", "ch"), ("cch", "chh"), ("ch", "chh"), ("sh", "s"), ("bh", "v"), ("v", "bh"), ("oo", "u"), ("u", "oo"),
        ("ee", "i"), ("i", "ee"), ("z", "j"), ("aa", "a"), ("ph", "f")]
def vary(text, rnd):
    # how people really type romanized Bengali/Hindi: pacche/pachhe/pache, khubbb, no punctuation, all lowercase
    words = text.split()
    idx = [i for i, w in enumerate(words) if w.isascii() and w.isalpha() and len(w) > 2]
    for i in rnd.sample(idx, min(len(idx), rnd.randint(1, 3))):
        w = words[i].lower()
        opts = [(a, b) for a, b in VARY if a in w]
        if opts:
            a, b = rnd.choice(opts); w = w.replace(a, b, 1)
        if rnd.random() < 0.15: w = w + w[-1] * 2
        words[i] = w
    t = " ".join(words)
    u = rnd.random()
    if u < 0.3: t = t.lower()
    elif u < 0.7: t = t[:1].upper() + t[1:]          # phone keyboards capitalise the first letter ("Haga pacche")
    if rnd.random() < 0.2: t = "".join(ch for ch in t if ch not in ".!?,")
    return t
def roman(text):
    return not any("\u0900" <= ch <= "\u09ff" for ch in text)
print(LX.find("haga pacche bhai", "bn"), lex_ids("haga pacche bhai", "bn"), mix([7], [112], [112, 1402]))
print(vary("Ami khub khushi, ghum pacche!", random.Random(1)))
""")

md("""
## 3. Training data

Every example says which outputs it teaches (`emo`, `tones`, `style`, `dims`); outputs it doesn't know about are masked out of the loss. Sources (all licences allow this use; non-commercial / unlicensed data is excluded):

| Source | Teaches | Licence |
|---|---|---|
| emojinize-multilingual | things/concepts → emojis, 14 languages incl. Hindi | CC-BY-4.0 |
| GoEmotions (27 emotions, Reddit) | feelings → tones + emojis (`goemotions_map.json`) | Apache-2.0 |
| Hinglish/English sarcasm headlines (BM2002) | sarcastic | CC-BY-4.0 |
| SemEval-2018 irony tweets (Natural Instructions task 386) | sarcastic | Apache-2.0 |
| Stanford politeness | respectful ↔ rude | MIT (orig. CC-BY-4.0) |
| Pavlick formality | formal ↔ casual | CC-BY-3.0 |
| Jigsaw toxicity | aggressive, rude, angry | Apache-2.0 |
| Unicode CLDR names/keywords | at least a few examples for every emoji | Unicode-3.0 |
| EmoJet seeds (hand-written, English/Hinglish/Banglish) | styles, hard tones, chat | ours |
| Synthetic replies | short drafts ("same", "lol", "yes!!") must follow the replied-to message | built here |
""")
code("""
from datasets import load_dataset
ex = []          # dict(text, reply, emo, tones, style, dims, src)
def add(text, src, reply=None, emo=None, tones=None, style=None, dims=None):
    text = strip_emojis(text)
    if text:
        ex.append(dict(text=text[:1200], reply=strip_emojis(reply)[:300] if reply else None, emo=emo or None,
                       tones=tones or None, style=style, dims=dims or None, src=src))

# emojinize: sentences (all emojis of their spans) + a sample of spans on their own
emz = load_dataset("yagizgencer/emojinize-multilingual", "all")
span_pool = []
for r in emz["train"]:
    ids_all = []
    for sp in r["spans"]:
        ids = emoji_ids(sp["emojis"])
        if ids:
            span_pool.append((sp["text"], ids))
            ids_all += [i for i in ids if i not in ids_all]
    if ids_all:
        add(r["sentence"], "emojinize", emo=ids_all)
random.shuffle(span_pool)
for t, ids in span_pool[:90_000]:
    add(t, "emojinize-span", emo=ids)

# GoEmotions -> tones + emojis
ge = load_dataset("google-research-datasets/go_emotions", "simplified")
GE_LABELS, GE = ge_map["labels"], ge_map["map"]
GE_TONES = sorted({TONE_IDX[t] for v in GE.values() for t in v["tones"]})
def ge_targets(labels):
    names = [GE_LABELS[i] for i in labels]
    pos = {TONE_IDX[t] for n in names for t in GE[n]["tones"]}
    emo = []
    for n in names:
        emo += [i for i in emoji_ids(GE[n]["emojis"]) if i not in emo]
    return {i: float(i in pos) for i in GE_TONES}, emo
for r in ge["train"]:
    tones, emo = ge_targets(r["labels"])
    add(r["text"], "goemotions", emo=emo, tones=tones)

# sarcasm: English + Hindi-English code-mixed headlines
import pandas as pd
from huggingface_hub import hf_hub_download
SARC = TONE_IDX["sarcastic"]
for f, col in [("training_data_english.xlsx", "headline"), ("fine-tuning_data_hinglish.xlsx", "cm_headline")]:
    df = pd.read_excel(hf_hub_download("BM2002/hinglish-sarcasm-dataset", f, repo_type="dataset"))
    for t, y in zip(df[col], df["is_sarcastic"]):
        add(str(t), "sarcasm-headlines", tones={SARC: float(y)})
sarc_test = pd.read_excel(hf_hub_download("BM2002/hinglish-sarcasm-dataset", "testing_data_hinglish.xlsx", repo_type="dataset"))

# irony tweets (Natural Instructions task 386 = SemEval-2018 task 3)
try:
    irony = load_dataset("Lots-of-LoRAs/task386_semeval_2018_task3_irony_detection")["train"]
    n0 = len(ex)
    for r in irony:
        m = re.search(r"Input:\\s*(.*?)\\s*Output:", r["input"], re.S)
        y = r["output"][0] if isinstance(r["output"], list) else r["output"]
        if m and y.strip().lower() in ("ironic", "not"):
            add(m.group(1), "irony-tweets", tones={SARC: float(y.strip().lower() == "ironic")})
    print("irony tweets:", len(ex) - n0)
except Exception as e:
    print("irony tweets skipped:", repr(e)[:200])

# politeness
pol = load_dataset("Cleanlab/stanford-politeness", data_files={"train": "fine-tuning/train.csv"})["train"]  # repo mixes CSV layouts
RESP, RUDE = TONE_IDX["respectful"], TONE_IDX["rude"]
for r in pol:
    c = r["completion"].strip().lower()
    if c in ("polite", "impolite", "neutral"):
        tones = {"polite": {RESP: 1.0, RUDE: 0.0}, "impolite": {RESP: 0.0, RUDE: 1.0}, "neutral": {RUDE: 0.0}}[c]
        add(r["prompt"].replace("Question:", "").strip(), "politeness", tones=tones,
            dims={2: {"polite": 0.1, "neutral": 0.4, "impolite": 0.9}[c]})

# formality (avg_score -3 informal .. +3 formal)
FORMAL = TONE_IDX["formal"]
for r in load_dataset("osyvokon/pavlick-formality-scores")["train"]:
    s = r["avg_score"]
    tones = {FORMAL: 1.0} if s > 1.0 else ({FORMAL: 0.0} if s < -1.0 else None)
    add(r["sentence"], "formality", tones=tones, dims={1: min(1.0, max(0.0, (3.0 - s) / 6.0))})

# toxicity
jig = load_dataset("tasksource/jigsaw_toxicity")["train"].shuffle(seed=SEED)
AGG, ANG = TONE_IDX["aggressive"], TONE_IDX["angry"]
n_clean = n_tox = 0
for r in jig:
    tox = r["toxic"] or r["severe_toxic"] or r["threat"]
    if tox and n_tox < 15_000:
        n_tox += 1
        add(r["comment_text"], "toxicity", emo=emoji_ids("😡🤬💢"),
            tones={AGG: float(bool(r["severe_toxic"] or r["threat"] or r["insult"])), RUDE: float(bool(r["obscene"] or r["insult"])), ANG: 1.0})
    elif not tox and n_clean < 10_000:
        n_clean += 1
        add(r["comment_text"], "toxicity", tones={AGG: 0.0, RUDE: 0.0, ANG: 0.0})
    if n_tox >= 15_000 and n_clean >= 10_000:
        break

# CLDR: name + keywords for every emoji, so none of the 1,923 is never seen
for i, e in enumerate(catalog):
    add(e["name"].replace("flag: ", "flag of "), "cldr", emo=[i])
    for k in e["kw"][:2]:
        add(k, "cldr", emo=[i])

# chat-object data: things mentioned inside casual chat -> the thing's emoji + a feeling emoji
OBJ_GROUPS = {"Food & Drink", "Animals & Nature", "Travel & Places", "Activities", "Objects"}
TEMPLATES = [
    ("just got a new {x}!!", "🎉🤩"), ("my {x} broke again", "😩💔"), ("look at this {x} lol", "😂👀"),
    ("can't stop thinking about {x}", "😍✨"), ("ugh the {x} again", "🙄😤"), ("we're getting {x} tonight", "🎉😋"),
    ("bhai {x} dekh", "👀😂"), ("ami {x} niye bose achi", "😅👀"), ("aaj {x} ka plan hai", "🎉🙌"),
    ("is that a {x}?", "👀😮"), ("I love {x} so much", "😍❤️"), ("no more {x} for me", "😩🙅"),
    ("watching the {x} with family", "😊❤️"), ("forgot my {x} at home", "🤦😩"),
]
FLAG_TEMPLATES = [("landed in {x} finally", "✈️🎉"), ("{x} won!!!", "🎉🔥"), ("missing {x} so much", "🥺❤️"), ("trip to {x} next month", "✈️🤩")]
n0 = len(ex)
for i, e in enumerate(catalog):
    if e["group"] in OBJ_GROUPS:
        words = [e["name"]] + [k for k in e["kw"][:2] if len(k) > 2]
        for k in range(4):
            tpl, feel = random.choice(TEMPLATES)
            add(tpl.format(x=random.choice(words)), "chat-objects", emo=[i] + emoji_ids(feel))
    elif e["sub"] == "country-flag":
        country = e["name"].replace("flag: ", "")
        for tpl, feel in random.sample(FLAG_TEMPLATES, 2):
            add(tpl.format(x=country), "chat-objects", emo=[i] + emoji_ids(feel))
print("chat-object examples:", len(ex) - n0)

# v2: real topic words (emojinize spans) wrapped in casual chat, so a topic keeps its emoji inside emotional sentences
if V2:
    CHAT_WRAP = ["{s} lol", "omg {s}", "bro {s} again", "talking about {s} rn", "can't stop thinking about {s}", "{s} tonight?",
                 "just saw {s}", "ugh {s}", "{s} is so good", "why is {s} like this", "kal {s} hai", "aaj {s} ka scene",
                 "bhai {s}", "ami {s} niye bhabchi", "finally {s}!!", "no more {s} please", "all day {s}", "{s} vibes",
                 "I'm into {s} these days", "need {s} asap", "{s} again lmao", "help, {s}", "who else loves {s}", "{s} > everything"]
    n0 = len(ex)
    for t, ids in span_pool[90_000:240_000]:
        add(random.choice(CHAT_WRAP).format(s=t.strip()), "topic-chat", emo=ids)
    for x in [x for x in ex if x["src"] == "chat-objects"]:
        add(x["text"], "chat-objects", emo=x["emo"])           # double the chat-object examples
    print("v2 topic-in-chat examples:", len(ex) - n0)

# real messages (emojet-data kernel): what people actually typed, the emojis they used are the labels
real_test = []
if REAL:
    import glob, html, zlib, gzip as _gzip
    rdir = os.path.dirname(sorted(glob.glob("/kaggle/input/**/real/train.jsonl.gz", recursive=True))[0])
    n0 = len(ex)
    real_rows = [json.loads(l) for l in _gzip.open(f"{rdir}/train.jsonl.gz", "rt", encoding="utf-8")]
    real_test = [json.loads(l) for l in _gzip.open(f"{rdir}/test.jsonl.gz", "rt", encoding="utf-8")]
    vrnd = random.Random(SEED)
    n_topic = 0
    for r in real_rows:
        text = html.unescape(r["t"])
        emo, emo_p = r["e"], None
        if ROUND3:
            topic = lex_ids(text, r["l"])
            if topic:
                n_topic += 1
                emo, emo_p = mix(r["e"], (), topic)
        for rep_ in range(LANG_REPEAT.get(r["l"], 1)):        # Bengali (and in round 2 Hindi) are scarcer: seen more often
            t = vary(text, vrnd) if (ROUND3 and rep_ and r["l"] in ("bn", "hi") and roman(text)) else text
            ex.append(dict(text=t, reply=None, emo=emo, emo_p=emo_p, tones=None, style=None, dims=None, src="real-" + r["l"]))
    if ROUND3:
        print(f"real messages with word-meaning emojis added: {n_topic:,}")
    # long drafts: 2-6 real messages of one language in a row; the last one counts most (it's what is being sent)
    by_lang = {}
    for r in real_rows:
        by_lang.setdefault(r["l"], []).append(r)
    for _ in range(80_000):
        lang = random.choice(["en", "hi", "bn"])
        if not by_lang.get(lang): continue
        parts = random.sample(by_lang[lang], random.randint(2, 6))
        emo = []
        for r in reversed(parts):
            emo += [i for i in r["e"] if i not in emo]
        ex.append(dict(text=" ".join(r["t"] for r in parts)[:1200], reply=None, emo=emo[:6], tones=None, style=None,
                       dims=None, src="real-long"))
    print(f"real examples: {len(ex) - n0:,} (test held out: {len(real_test):,})")

# human-labelled tone/style datasets (emojet-tones kernel): sarcasm, condescension, manipulation, aggression, empathy,
# emotions, politeness, dialogue acts, negotiation/support strategies (EN/HI/BN) + real WhatsApp messages with emojis
if REAL:
    n0 = len(ex)
    for path in sorted(glob.glob("/kaggle/input/**/tones/tones.jsonl.gz", recursive=True)):
        for line in _gzip.open(path, "rt", encoding="utf-8"):
            r = json.loads(line)
            tones = {TONE_IDX[t]: v for t, v in r["tones"].items() if t in TONE_IDX}
            for _ in range(2 if r["l"] in ("bn", "hi") else 1):       # Hindi/Bengali labels are rare: seen twice
                ex.append(dict(text=r["t"], reply=r.get("r"), emo=r.get("e") or None, tones=tones or None,
                               style=STYLE_IDX.get(r.get("style")), dims=None, src="tones-" + r["s"]))
    print(f"human-labelled tone/style examples: {len(ex) - n0:,}")

# annotated real messages (emojet-annotate kernel): full-taxonomy tones, style, sliders; emojis the writer would add
if REAL:
    ann_files = sorted(glob.glob("/kaggle/input/**/annot/annotations.jsonl", recursive=True))
    n0 = len(ex)
    ann_test = []
    for path in ann_files:
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            tones = {i: 0.3 for i in range(NT)}               # tones not picked: weak negatives (max 3 were picked)
            for t in r.get("tones", []):
                if t in TONE_IDX: tones[TONE_IDX[t]] = 1.0
            tone_t = {i: (1.0 if v == 1.0 else 0.0) for i, v in tones.items()}
            dims = {k: min(9, max(0, int(v))) / 9 for k, v in enumerate(r.get("axes", [])[:ND])}
            gold = r.get("e") or []
            silver = emoji_ids("".join(r.get("emojis", [])))
            text = html.unescape(r["t"])
            if ROUND3 and r.get("curated") and not LX.find(text, r["l"]):
                continue          # labelled for a word the current matcher no longer sees (e.g. Spanish "que se haga")
            if ROUND3:      # round 2 used gold OR silver: a "haga 😂" message never taught 💩. Now all of them, weighted.
                emo, emo_p = mix(gold, silver, lex_ids(text, r["l"]))
            else:
                emo, emo_p = (gold or silver) or None, None
            x = dict(text=text, reply=None, emo=emo, emo_p=emo_p, ew=(1.0 if gold else 0.5),
                     tones=tone_t, tone_w=tones, style=STYLE_IDX.get(r.get("style")), dims=dims, src="annot-" + r["l"])
            if zlib.crc32(r["t"].encode()) % 50 == 0:    # stable 2% hold-out (same messages in every run)
                ann_test.append(x)                            # 2% held out to measure tone/style agreement
            else:
                if ROUND3:
                    # round-3 labels (they carry word meanings) x3 for curated chat words, x2 others; a round-2 "slang"
                    # message only counts extra if it really contains a lexicon word (half of round 2's did not)
                    if "notes" in r:
                        reps = 3 if r.get("curated") else 2
                    else:
                        reps = 2 if (r.get("s", "").startswith("slang") and LX.find(text, r["l"])) else 1
                else:
                    reps = 3 if (ROUND2 and r.get("s", "").startswith("slang")) else 1
                for k_ in range(reps):
                    if ROUND3 and k_ and r["l"] in ("bn", "hi") and roman(text):
                        ex.append({**x, "text": vary(text, vrnd)})
                    else:
                        ex.append(x)
    print(f"annotated examples: {len(ex) - n0:,} (+{len(ann_test):,} held out)")
    if ROUND3:      # synthetic chat lines (emojet-annotate-3): silver, never a test message
        _probe = {c["text"].strip().lower() for k in ("chat_cases", "slang_probes", "slang_negatives", "emoji_cases", "tone_cases")
                  for c in evals.get(k, [])}
        n0 = len(ex); vrnd = random.Random(SEED + 1); dropped = 0
        for path in sorted(glob.glob("/kaggle/input/**/annot/synthetic.jsonl", recursive=True)):
            for line in open(path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                text = strip_emojis(r["t"])
                if text.strip().lower() in _probe:
                    dropped += 1; continue
                emo, emo_p = mix((), emoji_ids("".join(r.get("emojis", []))), lex_ids(text, r["l"]))
                if not emo:
                    continue
                for rep_ in range(2):
                    t = vary(text, vrnd) if (rep_ and roman(text)) else text
                    ex.append(dict(text=t, reply=None, emo=emo, emo_p=emo_p, ew=0.5, tones=None, style=None, dims=None,
                                   src="synth-" + r["l"]))
        print(f"synthetic chat examples: {len(ex) - n0:,} ({dropped} dropped: same text as a test message)")

# seeds (x25 with light variations)
def seed_targets(s):
    # hand-written: tones not listed are absent -> every tone gets negative examples too
    tones = {i: 0.0 for i in range(NT)}
    tones.update({TONE_IDX[t]: 1.0 for t in s.get("tones", []) if t in TONE_IDX})
    return emoji_ids(s["emojis"]), tones, STYLE_IDX.get(s.get("style"))
for s in seeds:
    emo, tones, style = seed_targets(s)
    for k in range(25):
        t = s["text"]
        if k % 3 == 1: t = t.lower()
        if k % 3 == 2: t = re.sub(r"[!?.]+$", "", t)
        add(t, "seeds", reply=s.get("reply"), emo=emo, tones=tones, style=style)

# synthetic replies: an uninformative draft takes its emojis from the replied-to message
GENERIC = ["same", "same here", "true", "fr", "exactly", "yes!!", "omg", "wow", "haha", "lol", "hahaha", "no way",
           "really?", "nice", "sahi hai", "ekdom", "haan", "accha", "seriously?", "ikr", "damn", "yesss", "sotti?", "uff"]
pool = [x for x in ex if x["src"] in ("goemotions", "seeds", "emojinize", "real-en", "real-hi", "real-bn") and x["emo"] and len(x["text"]) < 160]
for x in random.sample(pool, 15_000):
    add(random.choice(GENERIC), "reply-synth", reply=x["text"], emo=x["emo"], tones=x["tones"])
for x in random.sample(pool, 5_000):       # informative draft + unrelated reply: the draft wins
    y = random.choice(pool)
    add(x["text"], "reply-synth", reply=y["text"], emo=x["emo"], tones=x["tones"], style=x["style"])

if REPLAY:
    _r = random.Random(SEED + 2)
    ex = [x for x in ex if _r.random() < REPLAY.get(x["src"], 1.0)]
random.shuffle(ex)
if SMOKE:
    ex = ex[:4000]
from collections import Counter
print(f"{len(ex):,} training examples")
for src, n in Counter(x["src"] for x in ex).most_common():
    print(f"  {src:18s} {n:7,d}")
covered = {i for x in ex if x["emo"] for i in x["emo"]}
print(f"emojis with at least one example: {len(covered)}/{NE}")
print("examples:", [(x["text"][:50], [EMOJIS[i] for i in (x["emo"] or [])][:4]) for x in ex[:5]])
""")

md("## 4. Model: Laya's encoder + a new 1,977-output layer")
code("""
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, ModernBertForSequenceClassification

LAYA = snapshot_download("convaiinnovations/laya", revision=LAYA_REV,
                         allow_patterns=["encoder/*", "tokenizer/*", "model.safetensors", "*.json"])
tok = AutoTokenizer.from_pretrained(f"{LAYA}/tokenizer")
cfg = AutoConfig.from_pretrained(f"{LAYA}/encoder")
cfg.architectures = ["ModernBertForSequenceClassification"]
cfg.num_labels = N_LABELS
cfg.id2label = {i: l for i, l in enumerate(LABELS)}
cfg.label2id = {l: i for i, l in enumerate(LABELS)}
cfg.classifier_pooling = "mean"          # llama.cpp uses mean pooling for ModernBERT classifiers
cfg.problem_type = "multi_label_classification"
cfg.reference_compile = False
model = ModernBertForSequenceClassification(cfg)
sd = load_file(f"{LAYA}/model.safetensors")
res = model.model.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=False)
assert not res.missing_keys and not res.unexpected_keys, res
del sd
teacher = None
if RESUME or V2:
    import glob, copy
    tdir = os.path.dirname(sorted(glob.glob("/kaggle/input/**/emojet.json", recursive=True))[0])
    model = ModernBertForSequenceClassification.from_pretrained(tdir, torch_dtype=torch.float32)
    if model.config.num_labels != N_LABELS:
        # the label set grew: keep the encoder and every known label's output row, add the new ones
        old = model
        old_idx = {l: int(i) for i, l in old.config.id2label.items()}
        ncfg = copy.deepcopy(old.config)
        ncfg.num_labels = N_LABELS
        ncfg.id2label = {i: l for i, l in enumerate(LABELS)}
        ncfg.label2id = {l: i for i, l in enumerate(LABELS)}
        model = ModernBertForSequenceClassification(ncfg)
        osd, nsd = old.state_dict(), model.state_dict()
        for k in nsd:
            if not k.startswith("classifier."):
                nsd[k] = osd[k]
        kept_rows = 0
        for i, l in enumerate(LABELS):
            j = old_idx.get(l)
            if j is not None:
                nsd["classifier.weight"][i] = osd["classifier.weight"][j]
                nsd["classifier.bias"][i] = osd["classifier.bias"][j]
                kept_rows += 1
        model.load_state_dict(nsd)
        del old, osd
        print(f"label set grew {len(old_idx)} -> {N_LABELS}: {kept_rows} output rows kept, {N_LABELS - kept_rows} new")
    print(f"starting from {json.load(open(tdir + '/emojet.json')).get('name', 'Emojet')} at {tdir}")
if MINI:
    import glob, copy
    tdir = os.path.dirname(sorted(glob.glob("/kaggle/input/**/emojet.json", recursive=True))[0])
    teacher = ModernBertForSequenceClassification.from_pretrained(tdir, torch_dtype=torch.float16).cuda().eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    scfg = copy.deepcopy(teacher.config)
    scfg.num_hidden_layers = len(KEEP)
    if getattr(scfg, "layer_types", None):           # newer transformers validate one type per layer
        scfg.layer_types = [teacher.config.layer_types[k] for k in KEEP]
    model = ModernBertForSequenceClassification(scfg)
    tsd = {k: v.float() for k, v in teacher.state_dict().items()}
    ssd = {}
    for k in model.state_dict():
        if k.startswith("model.layers."):
            j = int(k.split(".")[2])
            ssd[k] = tsd[k.replace(f"model.layers.{j}.", f"model.layers.{KEEP[j]}.", 1)]
        else:
            ssd[k] = tsd[k]
    model.load_state_dict(ssd, strict=True)
    print(f"teacher Emojet-v1 from {tdir}; student keeps layers {KEEP}")
model.cuda()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters in {NAME}")
""")

md("## 5. Batches (similar lengths together = fast) and loss")
code("""
def encode(xs):
    enc = tok([fmt(x["text"], x["reply"]) for x in xs], truncation=True, max_length=MAX_LEN)
    return enc["input_ids"]
t0 = time.time()
ids_all = []
for i in range(0, len(ex), 20_000):
    ids_all += encode(ex[i:i + 20_000])
print(f"tokenized in {time.time() - t0:.0f} s; median length {int(np.median([len(i) for i in ids_all]))} tokens")

# emoji label frequencies (for the balanced loss)
_cnt = np.ones(NE)
for x in ex:
    for k, i in enumerate(x["emo"] or []):
        _cnt[i] += x["emo_p"][k] if x.get("emo_p") else 1.0 / len(x["emo"])
LOG_PRIOR = torch.tensor(np.log(_cnt / _cnt.sum()), dtype=torch.float32, device="cuda")
print("most frequent emojis:", "".join(EMOJIS[i] for i in np.argsort(-_cnt)[:20]))

MAX_TOKENS = 3584 if MAX_LEN > 64 else None          # 5120 ran out of memory on a T4 (15 GB) with 256-token texts
def make_batches(n, bs, rng):
    order = rng.permutation(n)
    chunks = [order[i:i + bs * 50] for i in range(0, n, bs * 50)]
    batches = []
    for ch in chunks:
        ch = sorted(ch, key=lambda j: len(ids_all[j]))
        if MAX_TOKENS is None:
            batches += [ch[i:i + bs] for i in range(0, len(ch), bs)]
            continue
        cur = []                                 # long texts: fewer per batch, so a batch never exceeds MAX_TOKENS
        for j in ch:
            if cur and (len(cur) == bs or (len(cur) + 1) * len(ids_all[j]) > MAX_TOKENS):
                batches.append(cur); cur = []
            cur.append(j)
        if cur: batches.append(cur)
    rng.shuffle(batches)
    return batches

def collate(idx):
    L = max(len(ids_all[j]) for j in idx)
    B = len(idx)
    inp = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    emo_t = torch.zeros((B, NE)); emo_m = torch.zeros(B, dtype=torch.bool); emo_w = torch.ones(B)
    tone_t = torch.zeros((B, NT)); tone_m = torch.zeros((B, NT))
    style = torch.full((B,), -1, dtype=torch.long)
    dim_t = torch.zeros((B, ND)); dim_m = torch.zeros((B, ND))
    for b, j in enumerate(idx):
        s = ids_all[j]; x = ex[j]
        inp[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        if x["emo"]:
            emo_t[b, x["emo"]] = torch.tensor(x["emo_p"]) if x.get("emo_p") else 1.0 / len(x["emo"])
            emo_m[b] = True; emo_w[b] = x.get("ew", 1.0)
        tw = x.get("tone_w") or {}
        for k, v in (x["tones"] or {}).items():
            tone_t[b, k] = v; tone_m[b, k] = tw.get(k, 1.0)
        if x["style"] is not None:
            style[b] = x["style"]
        for k, v in (x["dims"] or {}).items():
            dim_t[b, k] = v; dim_m[b, k] = 1
    return dict(inp=inp, att=att, emo_t=emo_t, emo_m=emo_m, emo_w=emo_w, tone_t=tone_t, tone_m=tone_m, style=style, dim_t=dim_t, dim_m=dim_m)

def loss_fn(logits, b):
    l = logits.float(); parts = {}; total = 0.0
    if b["emo_m"].any():   # several right emojis per message: soft-target cross-entropy over all 1,923
        le = l[b["emo_m"], :NE]
        if BALANCE_TAU:      # balanced softmax: train on logits + tau*log(prior) so the raw logits stay unbiased
            le = le + BALANCE_TAU * LOG_PRIOR
        w = b["emo_w"][b["emo_m"]]
        e = (-(b["emo_t"][b["emo_m"]] * torch.log_softmax(le, -1)).sum(-1) * w).sum() / w.sum()
        total = total + e; parts["emoji"] = e.item()
    if b["tone_m"].sum() > 0:
        t = (F.binary_cross_entropy_with_logits(l[:, OFF_T:OFF_S], b["tone_t"], reduction="none") * b["tone_m"]).sum() / b["tone_m"].sum()
        total = total + TONE_W * t; parts["tone"] = t.item()
    m = b["style"] >= 0
    if m.any():
        s = F.cross_entropy(l[m, OFF_S:OFF_D], b["style"][m])
        total = total + STYLE_W * s; parts["style"] = s.item()
    if b["dim_m"].sum() > 0:
        d = (F.binary_cross_entropy_with_logits(l[:, OFF_D:], b["dim_t"], reduction="none") * b["dim_m"]).sum() / b["dim_m"].sum()
        total = total + DIM_W * d; parts["dim"] = d.item()
    return total, parts
""")

md("## 6. Evaluation helpers (same test messages as the laptop tests; never trained on)")
code("""
@torch.no_grad()
def predict(drafts, replies=None, bs=128):
    model.eval()
    replies = replies or [None] * len(drafts)
    out = []
    for i in range(0, len(drafts), bs):
        enc = tok([fmt(d, r) for d, r in zip(drafts[i:i + bs], replies[i:i + bs])], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.float().cpu())
    model.train()
    return torch.cat(out)

def top_emojis(row, k=6): return [EMOJIS[i] for i in row[:NE].topk(k).indices.tolist()]
def top_tones(row, k=3):
    p = torch.sigmoid(row[OFF_T:OFF_S]); idx = p.topk(k).indices.tolist()
    return [(TONES[i], round(p[i].item(), 2)) for i in idx]
def style_of(row):
    p = torch.softmax(row[OFF_S:OFF_D], -1); i = int(p.argmax())
    return STYLES[i], round(p[i].item(), 2)

def evaluate(verbose=False):
    r = {}
    ec = evals["emoji_cases"]
    lg = predict([c["text"] for c in ec], [c["reply"] for c in ec])
    hits = 0
    for c, row in zip(ec, lg):
        top = top_emojis(row); ok = {norm(e) for e in c["ok_emojis"]}
        h = any(norm(e) in ok for e in top); hits += h
        if verbose: print(f"{'✓' if h else '✗'} {c['name']:18s} {''.join(top)}   {c['text'][:60]}")
    r["emoji_cases_hit@6"] = f"{hits}/{len(ec)}"
    tc = evals["tone_cases"]
    lg = predict([c["text"] for c in tc])
    th = sh = sn = 0
    for c, row in zip(tc, lg):
        tt = [t for t, _ in top_tones(row)]; st, _ = style_of(row)
        th += bool(set(tt) & set(c["ok_tones"]))
        if c["ok_styles"]: sn += 1; sh += st in c["ok_styles"]
        if verbose: print(f"{'✓' if set(tt) & set(c['ok_tones']) else '✗'} tones={tt} style={st:18s} {c['text'][:55]}")
    r["tone_top3"] = f"{th}/{len(tc)}"; r["style"] = f"{sh}/{sn}"
    sub = [x for x in emz["test"].select(range(min(1500, len(emz["test"]))))]
    gold = [set(emoji_ids("".join(sp["emojis"] for sp in x["spans"]))) for x in sub]
    lg = predict([strip_emojis(x["sentence"]) for x in sub])
    tops = [set(row[:NE].topk(6).indices.tolist()) for row in lg]
    r["emojinize_test_hit@6"] = round(float(np.mean([bool(t & g) for t, g in zip(tops, gold) if g])), 3)
    r["distinct_emojis_suggested"] = len(set().union(*tops))
    gt = [x for x in ge["test"].select(range(2000)) if 27 not in x["labels"]]
    lg = predict([x["text"] for x in gt])
    r["goemotions_emoji_hit@6"] = round(float(np.mean([bool(set(row[:NE].topk(6).indices.tolist()) & set(ge_targets(x["labels"])[1])) for x, row in zip(gt, lg)])), 3)
    if real_test:
        for lang in ("en", "hi", "bn"):
            rt = [x for x in real_test if x["l"] == lang]
            if not rt: continue
            lg = predict([x["t"] for x in rt])
            top6 = [row[:NE].topk(6).indices.tolist() for row in lg]
            r[f"real_{lang}_top1"] = round(float(np.mean([t[0] in x["e"] for t, x in zip(top6, rt)])), 3)
            r[f"real_{lang}_hit@6"] = round(float(np.mean([bool(set(t) & set(x["e"])) for t, x in zip(top6, rt)])), 3)
    cc = evals.get("chat_cases", [])
    if cc:
        lg = predict([c["text"] for c in cc], [c.get("reply") for c in cc])
        h1 = h3 = 0
        for c, row in zip(cc, lg):
            top = top_emojis(row); ok = {norm(e) for e in c["ok_emojis"]}
            h1 += norm(top[0]) in ok; h3 += any(norm(e) in ok for e in top[:3])
            if verbose: print(f"{'✓' if any(norm(e) in ok for e in top[:3]) else '✗'} {''.join(top)}   {c['text'][:60]}")
        r["chat_cases_top1"] = f"{h1}/{len(cc)}"; r["chat_cases_top3"] = f"{h3}/{len(cc)}"
    sp = evals.get("slang_probes", [])
    if sp:
        lg = predict([c["text"] for c in sp], [c.get("reply") for c in sp])
        h6 = h3 = 0; per = Counter(); tot = Counter()
        for c, row in zip(sp, lg):
            top = top_emojis(row); ok = {norm(e) for e in c["ok_emojis"]}
            a6 = any(norm(e) in ok for e in top); h6 += a6; h3 += any(norm(e) in ok for e in top[:3])
            per[c["lang"]] += a6; tot[c["lang"]] += 1
            if verbose: print(f"{'✓' if a6 else '✗'} slang {''.join(top)}   {c['text'][:50]}" + (f"  ↪ {c['reply']}" if c.get("reply") else ""))
        r["slang_hit@6"] = f"{h6}/{len(sp)}"; r["slang_hit@3"] = f"{h3}/{len(sp)}"
        lg2 = predict([c["text"][:1].upper() + c["text"][1:] for c in sp], [c.get("reply") for c in sp])
        hc = sum(any(norm(e) in {norm(x) for x in c["ok_emojis"]} for e in top_emojis(row)) for c, row in zip(sp, lg2))
        r["slang_capitalised_hit@6"] = f"{hc}/{len(sp)}"      # how phones send it: "Haga pacche"
        for l in sorted(tot): r[f"slang_{l}_hit@6"] = f"{per[l]}/{tot[l]}"
    sn = evals.get("slang_negatives", [])
    if sn:
        lg = predict([c["text"] for c in sn])
        clean = 0
        for c, row in zip(sn, lg):
            top = top_emojis(row); ok = not ({norm(e) for e in top} & {norm(e) for e in c["bad_emojis"]}); clean += ok
            if verbose: print(f"{'✓' if ok else '✗'} must not get {''.join(c['bad_emojis'])}: {''.join(top)}   {c['text']}")
        r["slang_negatives_clean"] = f"{clean}/{len(sn)}"
    lg = predict([str(t) for t in sarc_test["cm_headline"]])
    r["hinglish_sarcasm_acc"] = round(float(np.mean([(torch.sigmoid(row[OFF_T + SARC]) > 0.5).item() == bool(y) for row, y in zip(lg, sarc_test["is_sarcastic"])])), 3)
    return r
""")

md("## 7. Train")
code("""
head_params = [p for n, p in model.named_parameters() if not n.startswith("model.")]
enc_params = [p for n, p in model.named_parameters() if n.startswith("model.")]
opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}], weight_decay=0.01)
rng = np.random.RandomState(SEED)
steps_per_epoch = len(make_batches(len(ex), BATCH, np.random.RandomState(0)))
total_steps = steps_per_epoch * EPOCHS
warm = int(WARMUP * total_steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total_steps - s) / max(1, total_steps - warm)))
FIT_AT = 300      # after this many steps: if the run would not finish in time, shorten it so the LR still decays to 0
scaler = torch.amp.GradScaler("cuda")
history, step, t_train = [], 0, time.time()
baseline = None
if V2 or RESUME:            # continuing a trained model: measure where it starts, so the end result can be compared
    baseline = evaluate(); print("starting model:", baseline, flush=True)
t_train = time.time()       # after that evaluation: round 3 counted it as training time and planned 3x too few steps
n_oom, crash = 0, None
print(f"{len(ex):,} examples, {steps_per_epoch:,} steps/epoch, {EPOCHS} epochs")
model.train()
stop = False
for epoch in range(EPOCHS):
    run = Counter(); cnt = Counter()
    batches = make_batches(len(ex), BATCH, rng)
    if epoch == 0:          # the biggest batch first: if memory is too tight, fail in the first minute, not hours in
        big = max(range(len(batches)), key=lambda i: len(batches[i]) * max(len(ids_all[j]) for j in batches[i]))
        batches.insert(0, batches.pop(big))
    for bi, idx in enumerate(batches):
      try:
        b = {k: v.cuda(non_blocking=True) for k, v in collate(idx).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=b["inp"], attention_mask=b["att"]).logits
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(input_ids=b["inp"], attention_mask=b["att"]).logits.float()
        loss, parts = loss_fn(logits, b)
        if teacher is not None:          # learn the teacher's full answer on every example
            sl = logits.float()
            kd = F.kl_div(torch.log_softmax(sl[:, :NE] / KD_T, -1), torch.softmax(tl[:, :NE] / KD_T, -1), reduction="batchmean") * KD_T ** 2
            kd = kd + F.binary_cross_entropy_with_logits(sl[:, OFF_T:OFF_S], torch.sigmoid(tl[:, OFF_T:OFF_S]))
            kd = kd + F.kl_div(torch.log_softmax(sl[:, OFF_S:OFF_D], -1), torch.softmax(tl[:, OFF_S:OFF_D], -1), reduction="batchmean")
            kd = kd + F.binary_cross_entropy_with_logits(sl[:, OFF_D:], torch.sigmoid(tl[:, OFF_D:]))
            loss = SUP_W * loss + KD_W * kd
            parts["kd"] = kd.item()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        for k, v in parts.items(): run[k] += v; cnt[k] += 1
        if step == FIT_AT:
            per_step = (time.time() - t_train) / FIT_AT
            left_h = TIME_BUDGET_H - (time.time() - T_START) / 3600 - 0.25       # 15 min for evaluation + saving
            fit = int(left_h * 3600 / per_step) + FIT_AT
            if fit < total_steps:
                print(f"fitting the run to the time budget: {total_steps:,} -> {fit:,} steps ({fit / steps_per_epoch:.2f} epochs)")
                total_steps = fit
        if step >= total_steps:
            stop = True; break
        if step % 500 == 0:
            avg = {k: round(run[k] / cnt[k], 4) for k in run}
            history.append(dict(step=step, epoch=epoch, **avg))
            el = time.time() - t_train
            print(f"epoch {epoch + 1} step {step:,}/{total_steps:,}  {avg}  {el / 60:.0f} min, ~{el / step * (total_steps - step) / 60:.0f} min left", flush=True)
            run.clear(); cnt.clear()
        if ROUND3 and step == total_steps // 2:
            print("--- halfway:", evaluate(), flush=True)
        if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
            print("time budget reached, stopping early"); stop = True; break
      except torch.cuda.OutOfMemoryError:
        # one unusually long batch: skip it rather than lose the run (the biggest batch ran first, so this is rare)
        n_oom += 1; opt.zero_grad(set_to_none=True); b = logits = loss = None; torch.cuda.empty_cache()
        print(f"out of memory on a batch of {len(idx)}, skipped ({n_oom})", flush=True)
        if n_oom > 20: crash = "out of memory 20 times"; stop = True; break
      except Exception as exc:
        # anything else: stop training but still evaluate and save what was learned (hours of GPU time)
        import traceback; traceback.print_exc(); crash = repr(exc)[:300]; stop = True; break
    print(f"--- after epoch {epoch + 1}:", evaluate(), flush=True) if not crash else None
    if stop: break
print(f"training took {(time.time() - t_train) / 60:.0f} min")
""")

md("## 8. Results")
code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4))
for key in ("emoji", "tone", "style", "dim", "kd"):
    pts = [(h["step"], h[key]) for h in history if key in h]
    if pts: ax.plot(*zip(*pts), label=key)
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend(); ax.set_title(f"{NAME} training loss")
plt.tight_layout(); plt.savefig("/kaggle/working/loss.png", dpi=120); plt.show()
final = evaluate(verbose=True)
print(json.dumps(final, indent=1))
if crash: print("!! training stopped early:", crash)
try:                        # what got better and what got worse than the starting model (never blocks the save below)
    if baseline:
        def _num(v):
            if isinstance(v, str) and "/" in v:
                a, b_ = v.split("/"); return float(a) / max(1.0, float(b_))
            return float(v)
        ref = REFERENCE or baseline
        print(f"{'metric':28s} {'round 2':>9s} {'start':>9s} {'now':>9s}   (flags: now vs round 2, the shipped model)")
        for k in final:
            a = _num(ref[k]) if k in ref else None
            b_ = _num(final[k])
            flag = "" if a is None else ("  ↑" if b_ > a + 1e-9 else ("  ↓ WORSE" if b_ < a - 1e-9 else ""))
            print(f"{k:28s} {str(ref.get(k, '-')):>9s} {str(baseline.get(k, '-')):>9s} {str(final[k]):>9s}{flag}")
except Exception as exc:
    print("comparison table failed:", repr(exc))
""")

md("## 9. Try it — edit the sentences and run this cell")
code("""
def demo(text, reply=None):
    row = predict([text], [reply])[0]
    print(("↪ " + reply + "\\n") if reply else "", text)
    print("   emojis:", " ".join(top_emojis(row, 8)))
    print("   tones :", top_tones(row), "| style:", style_of(row))
for t in ["Bhai ami khete jacchi, pore kotha bolchi", "Happy birthday bro!! have an amazing year",
          "oh great, another monday meeting that could have been an email", "Fine, I'll do it your way, since you always know best.",
          "landed in Goa finally, beach time", "India won the final!!!", "it's fine. really. I'm fine."]:
    demo(t)
demo("yes!!", reply="do you want to come to the concert with me?")
demo("same", reply="I'm so done with this week")
""")

md("## 10. Save: Hugging Face weights, labels, reference outputs, and the GGUF file the phone runs")
code("""
OUT = f"/kaggle/working/{NAME}"
model.eval()
ref_texts = [("Happy birthday bro!! have an amazing year", None), ("Bhai ami khete jacchi, pore kotha bolchi", None),
             ("yes!!", "do you want to come to the concert with me?"), ("it's fine. really. I'm fine.", None)]
ref = []
for d, r in ref_texts:
    enc = tok(fmt(d, r), return_tensors="pt").to("cuda")
    with torch.no_grad():
        row = model.float()(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits[0].cpu()
    ref.append(dict(text=fmt(d, r), input_ids=enc["input_ids"][0].tolist(), logits=[round(v, 4) for v in row.tolist()],
                    top_emojis=top_emojis(row), top_tones=top_tones(row), style=style_of(row)))
model.half().save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
meta = dict(name=NAME, version=1, base="convaiinnovations/laya@" + LAYA_REV, labels=LABELS,
            groups=dict(emojis=[0, NE], tones=[OFF_T, OFF_S], styles=[OFF_S, OFF_D], dims=[OFF_D, N_LABELS]),
            input_format="replying to: {reply}\\nmy message: {draft}  (first line only when replying)",
            max_len=MAX_LEN, pooling="mean", metrics=final, baseline=baseline, crash=crash, round=(3 if ROUND3 else (2 if ROUND2 else 1)),
            data_commit=COMMIT, history=history)
json.dump(meta, open(f"{OUT}/emojet.json", "w"), ensure_ascii=False)
json.dump(ref, open("/kaggle/working/reference.json", "w"), ensure_ascii=False)
print(os.listdir(OUT))

# GGUF for llama.cpp (same llama.cpp commit as the phone engine)
try:
    subprocess.run(["git", "clone", "-q", "https://github.com/ggml-org/llama.cpp", "/tmp/llama.cpp"], check=True)
    subprocess.run(["git", "-C", "/tmp/llama.cpp", "checkout", "-q", LLAMA_CPP_COMMIT], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "/tmp/llama.cpp/gguf-py"], check=True)
    r = subprocess.run([sys.executable, "/tmp/llama.cpp/convert_hf_to_gguf.py", OUT, "--outtype", "f16",
                        "--outfile", f"/kaggle/working/{NAME}-f16.gguf"], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-2500:])
except Exception as e:
    print("GGUF conversion failed (weights are saved; convert later):", repr(e))
print(f"total notebook time {(time.time() - T_START) / 60:.0f} min")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "training/kaggle" / (("emojet_v1_mini" if MINI else ("emojet_v2_round3" if ROUND3 else "emojet_v2_round2" if ROUND2 else "emojet_v2_real" if REAL else "emojet_v2" if V2 else ("emojet_v1_resume" if RESUME else "emojet_v1"))) + ("_smoke" if SMOKE else "") + ".ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells, data commit {COMMIT[:10]})")
