"""Emojet human-labelled tone/style data (runs on Kaggle, CPU, internet on).

Collects public datasets where PEOPLE labelled tone, emotion, sarcasm, condescension, manipulation,
aggression, politeness, dialogue acts or negotiation/support strategies, and maps every label onto
the taxonomy in labels/tones.json (65 tones, 8 styles). Plus real WhatsApp group messages with emojis.

Output /kaggle/working/tones/tones.jsonl.gz:
  {"t": text, "r": message replied to or null, "l": "en"|"hi"|"bn", "s": source,
   "tones": {tone: 1.0 | 0.0}, "style": style or null, "e": [emoji ids]}
A tone missing from "tones" is unknown for that row (not trained), 0.0 = labelled absent.

Built by training/build_tone_data.py (fills in TAXONOMY_B64 and SMOKE).
"""
import base64
import collections
import csv
import glob
import gzip
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

SMOKE = __SMOKE__
TAXONOMY_B64 = "__TAXONOMY__"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "emoji==2.14.1"], check=False)
import emoji  # noqa: E402
from datasets import load_dataset  # noqa: E402

T0 = time.time()
random.seed(11)
csv.field_size_limit(10_000_000)
OUT = "/kaggle/working/tones"
os.makedirs(OUT, exist_ok=True)
TAX = json.loads(gzip.decompress(base64.b64decode(TAXONOMY_B64)).decode())
TONES, STYLES, EMOJIS = set(TAX["tones"]), set(TAX["styles"]["options"]), TAX["emojis"]
SKIN = {chr(c) for c in range(0x1F3FB, 0x1F400)}
E_INDEX = {}
for i, e in enumerate(EMOJIS):
    E_INDEX.setdefault("".join(ch for ch in e if ch not in SKIN and ch != "️"), i)
CAP = 300 if SMOKE else None
rows, stats = [], collections.Counter()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def clean(t):
    t = re.sub(r"(https?://\S+|www\.\S+)", " ", str(t or ""))
    t = re.sub(r"(^|\s)@\w+", " ", t)
    return re.sub(r"\s+", " ", emoji.replace_emoji(t, " ")).strip()


def emoji_labels(t):
    out = []
    for m in emoji.emoji_list(t or ""):
        i = E_INDEX.get("".join(ch for ch in m["emoji"] if ch not in SKIN and ch != "️"))
        if i is not None and i not in out:
            out.append(i)
    return out[:6]


def add(src, text, lang, tones=None, style=None, reply=None, e=None):
    text = clean(text)
    if not 2 <= len(text) <= 1500:
        return
    tones = {k: float(v) for k, v in (tones or {}).items()}
    bad = [k for k in tones if k not in TONES] + ([style] if style and style not in STYLES else [])
    assert not bad, (src, bad)                        # a mapping typo must fail loudly, not train garbage
    if not tones and not style and not e:
        return
    rows.append({"t": text, "r": clean(reply)[:500] or None if reply else None, "l": lang, "s": src,
                 "tones": tones, "style": style, "e": e or []})
    stats[src] += 1


def capped(it, n):
    for k, x in enumerate(it):
        if (CAP and k >= CAP) or (n and k >= n):
            return
        yield x


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "emojet"})
    return urllib.request.urlopen(req, timeout=300).read()


def run(name, fn):
    t, n0 = time.time(), len(rows)
    try:
        fn()
    except Exception as exc:
        log(f"!! {name} failed: {exc!r}")
    log(f"{name}: +{len(rows) - n0:,} rows ({time.time() - t:.0f} s)")


def hf(repo, config=None, split="train"):
    return load_dataset(repo, config, split=split)


def on(*names):
    return {n: 1.0 for n in names}


GH = "https://raw.githubusercontent.com"


# ---------------------------------------------------------------- English
def isarcasm():
    data = fetch(f"{GH}/iabufarha/iSarcasmEval/main/train/train.En.csv").decode("utf-8", "replace")
    for r in capped(csv.DictReader(io.StringIO(data)), None):
        if r.get("sarcastic") == "1":
            t = on("sarcastic")
            if r.get("satire") == "1" or r.get("understatement") == "1":
                t["facetious"] = 1.0
            add("isarcasm", r["tweet"], "en", t)
        elif r.get("sarcastic") == "0":
            add("isarcasm", r["tweet"], "en", {"sarcastic": 0.0})


def sarc():
    ds = hf("marcbishara/sarcasm-on-reddit", split="sft_train").shuffle(seed=1)
    for r in capped(ds, 60_000):
        add("sarc", r["comment"], "en", {"sarcastic": float(r["label"] == 1)}, reply=r["parent_comment"])


def talkdown():
    tar = tarfile.open(fileobj=io.BytesIO(fetch("https://nlp.stanford.edu/~zijwang/talkdown/talkdown.tar.gz")))
    for name in ("balanced_train.jsonl", "balanced_dev.jsonl"):
        for line in capped(tar.extractfile(name).read().decode("utf-8", "replace").splitlines(), None):
            r = json.loads(line)
            yes = str(r.get("label")) == "True"
            add("talkdown", r["quotedpost"], "en", {"condescending": float(yes)})


MM_TECH = {"Accusation": ["accusatory"], "Denial": ["defiant", "defensive"], "Evasion": [], "Feigning Innocence": ["defensive"],
           "Rationalization": ["defensive"], "Playing Victim Role": ["defensive"], "Playing Servant Role": ["obsequious"],
           "Shaming or Belittlement": ["derisive", "condescending"], "Intimidation": ["aggressive", "authoritative"],
           "Brandishing Anger": ["angry", "aggressive"], "Persuasion or Seduction": ["flirty"]}


def mentalmanip():
    for r in capped(hf("audreyeleven/MentalManip", "mentalmanip_detailed"), None):
        votes = [r.get(f"manipulative_{k}") for k in (1, 2, 3)]
        manip = sum(int(v or 0) for v in votes) >= 2
        techs = collections.Counter(t.strip() for k in (1, 2, 3) for t in str(r.get(f"technique_{k}") or "").split(",") if t.strip())
        tones, style = {}, ("manipulative" if manip else None)
        for tech, n in techs.items():
            if n >= 2:
                for tone in MM_TECH.get(tech, []):
                    tones[tone] = 1.0
                if tech == "Evasion":
                    style = "avoidant"
        if manip and sum(techs.values()) and "machiavellian" not in tones and techs.get("Persuasion or Seduction", 0) + techs.get("Rationalization", 0) >= 2:
            tones["machiavellian"] = 1.0
        add("mentalmanip", r["dialogue"], "en", tones, style)


CASINO = {"promote-coordination": ("collaborative", []), "empathy-coordination": ("collaborative", ["empathetic"]),
          "showing-empathy": (None, ["empathetic"]), "small-talk": (None, ["friendly"]), "self-need": ("transactional", []),
          "other-need": ("transactional", []), "no-need": ("transactional", []), "elicit-pref": (None, ["curious"]),
          "vouch-fair": ("assertive", []), "uv-part": ("manipulative", [])}


def casino():
    for r in capped(hf("kchawla123/casino"), None):
        prev = None
        for text, labs in r.get("annotations") or []:
            styles, tones = [], {}
            for lab in str(labs).split(","):
                st, ts = CASINO.get(lab.strip(), (None, []))
                if st: styles.append(st)
                for t in ts: tones[t] = 1.0
            style = collections.Counter(styles).most_common(1)[0][0] if styles else None
            add("casino", text, "en", tones, style, reply=prev)
            prev = text


ESC_STRAT = {"Affirmation and Reassurance": ["reassuring"], "Reflection of feelings": ["empathetic"],
             "Self-disclosure": ["sincere"], "Providing Suggestions": ["encouraging"], "Question": ["curious"]}
ESC_EMO = {"anxiety": ["anxious"], "depression": ["sad", "resigned"], "sadness": ["sad"], "anger": ["angry"],
           "fear": ["anxious"], "shame": ["embarrassed"], "disgust": ["disapproving"], "nervousness": ["anxious"],
           "pain": ["sad"], "guilt": ["apologetic"], "jealousy": ["bitter"]}


def esconv():
    for r in capped(hf("thu-coai/esconv"), None):
        d = json.loads(r["text"])
        prev = None
        for turn in d["dialog"]:
            if turn["speaker"] == "sys" and turn.get("strategy") in ESC_STRAT:
                add("esconv", turn["text"], "en", on(*ESC_STRAT[turn["strategy"]]), reply=prev)
            elif turn["speaker"] == "usr" and len(turn["text"]) > 25:
                add("esconv", turn["text"], "en", on(*ESC_EMO.get(d.get("emotion_type"), [])))
            prev = turn["text"]


ED = {"nostalgic": ["nostalgic"], "sentimental": ["nostalgic", "loving"], "caring": ["empathetic", "loving"],
      "grateful": ["appreciative"], "apprehensive": ["anxious"], "anxious": ["anxious"], "terrified": ["anxious"],
      "afraid": ["anxious"], "guilty": ["apologetic"], "ashamed": ["embarrassed"], "embarrassed": ["embarrassed"],
      "disappointed": ["disappointed"], "lonely": ["sad"], "hopeful": ["optimistic"], "proud": ["confident", "joyful"],
      "jealous": ["bitter"], "devastated": ["sad", "resigned"], "content": ["joyful"], "impressed": ["admiring"],
      "excited": ["enthusiastic"], "furious": ["angry"], "angry": ["angry"], "annoyed": ["angry"],
      "disgusted": ["disapproving"], "surprised": ["surprised"], "joyful": ["joyful"], "trusting": ["sincere"],
      "faithful": ["sincere"], "confident": ["confident"], "prepared": ["confident"], "anticipating": ["enthusiastic"],
      "sad": ["sad"]}


def empathetic_dialogues():
    for r in capped(hf("Estwld/empathetic_dialogues_llm"), None):
        conv = r["conversations"]
        if conv and conv[0]["role"] == "user":
            add("empathetic_dialogues", conv[0]["content"], "en", on(*ED.get(r["emotion"], [])))
        if len(conv) > 1 and conv[1]["role"] == "assistant":
            add("empathetic_dialogues", conv[1]["content"], "en", {"empathetic": 1.0}, reply=conv[0]["content"])


SE18 = {"anger": "angry", "disgust": "disapproving", "fear": "anxious", "joy": "joyful", "love": "loving",
        "optimism": "optimistic", "pessimism": "pessimistic", "sadness": "sad", "surprise": "surprised", "trust": "sincere",
        "anticipation": "enthusiastic"}


def semeval18():
    for r in capped(hf("akkasi/sem_eval_2018"), None):
        idx = {int(k): v for k, v in eval(r["idx2label"]).items()}   # dataset stores the mapping as a Python dict string
        add("semeval18", r["text"], "en", {SE18[idx[i]]: float(v) for i, v in enumerate(r["labels"]) if idx[i] in SE18})


def hate_measure():
    ds = hf("ucberkeley-dlab/measuring-hate-speech")
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    text = {}
    for r in ds:
        c = r["comment_id"]; text[c] = r["text"]
        for k in ("respect", "insult", "humiliate", "status", "sentiment"):
            agg[c][k].append(r[k] or 0)
    for c in capped(list(agg), 30_000):
        m = {k: sum(v) / len(v) for k, v in agg[c].items()}            # 0..4 Likert, mean over raters
        tones = {"derisive": float(m["humiliate"] >= 2.5), "condescending": float(m["status"] >= 2.5),
                 "rude": float(m["insult"] >= 2.5)}
        if m["respect"] <= 0.5 and m["insult"] <= 0.5:
            tones["respectful"] = 1.0
        add("measuring_hate", text[c], "en", tones)


def politeness():
    base = "https://huggingface.co/datasets/Cleanlab/stanford-politeness/resolve/main/fine-tuning/"
    for fname in ("train.csv", "test.csv"):           # the repo mixes CSV layouts: read these two files directly
        for r in capped(csv.DictReader(io.StringIO(fetch(base + fname).decode("utf-8", "replace"))), None):
            q = r["prompt"].replace("Question:", "").strip()
            lab = r["completion"].strip()
            if lab == "polite":
                add("politeness", q, "en", {"respectful": 1.0, "rude": 0.0})
            elif lab == "impolite":
                add("politeness", q, "en", {"rude": 1.0, "respectful": 0.0})


SWDA = {"fa": {"apologetic": 1.0}, "ft": {"appreciative": 1.0}, "h": {"tentative": 1.0}, "ad": {"demanding": 1.0},
        "ba": {"appreciative": 1.0, "admiring": 1.0}, "qw": {"curious": 1.0}, "qy": {"curious": 1.0}, "qo": {"curious": 1.0}}


def swda():
    ds = hf("nico8771/swda_processed_v2")
    names = ds.features["act_tag"].names
    for r in capped(ds, None):
        tag = names[r["act_tag"]]
        if tag in SWDA and len(r["text"]) > 12:
            add("swda", r["text"], "en", SWDA[tag])


def dreaddit():
    for r in capped(hf("andreagasparini/dreaddit"), None):
        add("dreaddit", r["text"], "en", {"anxious": float(r["label"] == 1)})


DD_EMO = {"anger": "angry", "disgust": "disapproving", "fear": "anxious", "happiness": "joyful", "sadness": "sad",
          "surprise": "surprised"}


def dailydialog():
    z = zipfile.ZipFile(io.BytesIO(fetch("https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip")))
    dialogs = json.loads(z.read("data/dialogues.json"))
    neutral = 0
    for d in capped(dialogs, None):
        prev = None
        for turn in d["turns"]:
            emo = turn.get("emotion")
            acts = {a["intent"] for a in turn.get("dialogue_acts", {}).get("binary", [])}
            tones = {DD_EMO[emo]: 1.0} if emo in DD_EMO else {}
            if "question" in acts:
                tones["curious"] = 1.0
            if emo == "no emotion" and not tones and neutral < 15_000:
                tones["neutral"] = 1.0; neutral += 1
            add("dailydialog", turn["utterance"], "en", tones, reply=prev)
            prev = turn["utterance"]


# ---------------------------------------------------------------- Hindi / Hinglish / Bengali
def trac2():
    for lang_dir, lang in (("IBEN", "bn"), ("HIN", "hi"), ("ENG", "en")):
        for split in ("train", "dev"):
            data = fetch(f"{GH}/socialmediaie/TRAC2020/master/{lang_dir}/Sub-task%20A/{split}.tsv").decode("utf-8", "replace")
            for cols in capped(csv.reader(io.StringIO(data), delimiter="\t"), None):
                lab = next((c for c in cols if c in ("NAG", "CAG", "OAG")), None)
                if not lab:
                    continue
                text = max(cols, key=len)
                if lab == "OAG":
                    add("trac2", text, lang, {"aggressive": 1.0}, "aggressive")
                elif lab == "CAG":
                    add("trac2", text, lang, {}, "passive-aggressive")          # covert aggression
                else:
                    add("trac2", text, lang, {"aggressive": 0.0})


def hindi_sarcasm():
    base = f"{GH}/pragyakatyayan/Tweets_Dataset_for_Sarcasm_detection_in_Hindi/master/"
    for fname, y in (("Sarcasm_Hindi_Tweets-SARCASTIC.csv", 1.0), ("Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv", 0.0)):
        data = fetch(base + fname).decode("utf-8", "replace")
        for r in capped(csv.DictReader(io.StringIO(data)), 6000):
            add("hindi_sarcasm", re.sub(r"#\w+", " ", r.get("text") or ""), "hi", {"sarcastic": y})


def hinglish_sarcasm():
    base = f"{GH}/sahilswami96/SarcasmDetection_CodeMixed/master/Dataset/"
    tw = fetch(base + "Sarcasm_tweets.txt").decode("utf-8", "replace").split("\n\n")
    truth = [l.strip() for l in fetch(base + "Sarcasm_tweet_truth.txt").decode("utf-8", "replace").splitlines() if l.strip()]
    labels = dict(zip(truth[0::2], truth[1::2]))      # id line, YES/NO line
    for block in capped(tw, None):
        p = block.strip().split("\n", 1)
        if len(p) == 2 and p[0].strip() in labels:
            add("hinglish_sarcasm", p[1], "hi", {"sarcastic": float(labels[p[0].strip()] == "YES")})


def kaggle_csv(pattern, src, lang, fn):
    for path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if re.search(pattern, path, re.I):
            with open(path, encoding="utf-8", errors="replace", newline="") as fh:
                rd = csv.DictReader(fh)
                cols = rd.fieldnames or []
                log(f"  {src}: {path} columns {cols}")
                for r in capped(rd, None):
                    fn(r, cols)


def text_col(r, cols):
    return max(((c, r.get(c) or "") for c in cols), key=lambda kv: len(kv[1]))[1]


def ben_sarc():
    import pandas as pd
    for path in glob.glob("/kaggle/input/**/*Ben-Sarc*.xlsx", recursive=True):
        df = pd.read_excel(path)
        log("  ben_sarc columns", list(df.columns))
        lab = next(c for c in df.columns if re.search(r"polarity|label|class|sarc", str(c), re.I) and set(df[c].dropna().astype(int).unique()) <= {0, 1})
        txt = max((c for c in df.columns if c != lab and df[c].dtype == object), key=lambda c: df[c].astype(str).str.len().mean())
        for _, r in capped(df.iterrows(), None):
            add("ben_sarc", r[txt], "bn", {"sarcastic": float(int(r[lab]) == 1)})


def bengali_humor():
    def one(r, cols):
        lab = next((r[c] for c in cols if c and re.search(r"humou?r|label|class", c, re.I) and str(r[c]).strip() in ("0", "1")), None)
        if lab is not None:
            add("bengali_humor", text_col(r, cols), "bn", {"humorous": float(lab == "1")})
    kaggle_csv(r"humou?r", "bengali_humor", "bn", one)


def bad_bengali():
    for split in ("train", "validation"):
        for r in capped(hf("omar-sharif/BAD-Bengali-Aggressive-Text-Dataset", split=split), None):
            ag = r["Class"] == "AG"
            add("bad_bengali", r["cleaned"], "bn", {"aggressive": float(ag)}, "aggressive" if ag else None)


CYPHER = {"troll": ["derisive"], "sexual": ["rude"], "religious": ["aggressive"], "threat": ["aggressive"]}


def bangla_fb():
    data = fetch(f"{GH}/cypher-07/Bangla-Text-Dataset/main/dataset.csv").decode("utf-8", "replace")
    for r in capped(csv.DictReader(io.StringIO(data)), None):
        lab = (r.get("label") or "").strip()
        if lab == "not bully":
            add("bangla_fb", r["comment"], "bn", {"rude": 0.0, "aggressive": 0.0, "derisive": 0.0})
        elif lab in CYPHER:
            add("bangla_fb", r["comment"], "bn", on(*CYPHER[lab]))


EMONOBA = {"Love": "loving", "Joy": "joyful", "Surprise": "surprised", "Anger": "angry", "Sadness": "sad", "Fear": "anxious"}


def emonoba():
    for split in ("train",):
        ds = hf("khondoker/EmoNoBa", split=split)
        for r in capped(ds, None):
            text = r.get("Data") or r.get("text") or ""
            add("emonoba", text, "bn", {t: float(bool(r.get(k))) for k, t in EMONOBA.items() if k in r})


def bengali_80k():
    import pandas as pd
    raw = fetch("https://data.mendeley.com/public-files/datasets/4dnrwbxt8n/files/cad4b87e-7167-4d2c-a849-5a900e528e3a/file_downloaded")
    df = pd.read_excel(io.BytesIO(raw))
    log("  bengali_80k columns", list(df.columns))
    col = next(c for c in df.columns if re.search("bengali|bangla", str(c), re.I) and not re.search("banglish", str(c), re.I))
    lab = next(c for c in df.columns if re.search("label|emotion", str(c), re.I))
    m = {"joy": "joyful", "sadness": "sad", "anger": "angry", "disgust": "disapproving", "surprise": "surprised", "fear": "anxious"}
    for _, r in capped(df.iterrows(), None):
        t = m.get(str(r[lab]).strip().lower())
        if t:
            add("bengali_80k", r[col], "bn", {t: 1.0})


ROMAN_HI = set("hai hain nahi nhi kya kyu bhai yaar mera meri tera teri accha acha bahut bohot kuch karo raha rahi gaya hoga "
               "tum aap mujhe sahi ekdum matlab kaise abhi wala ji haan bilkul pyaar dost ghar".split())


def whatsapp():
    z = zipfile.ZipFile(io.BytesIO(fetch(
        "https://data.mendeley.com/public-files/datasets/wf5d5b2j52/files/06911351-d087-4677-af89-c483136ffe03/file_downloaded")))
    name = next(n for n in z.namelist() if n.endswith(".db"))
    path = "/tmp/whatsapp.db"
    open(path, "wb").write(z.read(name))
    con = sqlite3.connect(path)
    table = next(r[0] for r in con.execute("select name from sqlite_master where type='table'") if "CONV" in r[0].upper())
    cols = [r[1] for r in con.execute(f"pragma table_info({table})")]
    tcol = next(c for c in cols if "TEXT" in c.upper() or "MSG" in c.upper())
    seen = set()
    for (msg,) in capped(con.execute(f"select {tcol} from {table}"), None):
        msg = str(msg or "")
        if "http" in msg or len(msg) > 400 or msg in seen:        # forwards and links are mostly spam
            continue
        seen.add(msg)
        e = emoji_labels(msg)
        if e:
            words = re.findall(r"[a-z]+", msg.lower())
            add("whatsapp", msg, "hi" if sum(w in ROMAN_HI for w in words) >= 1 else "en", e=e)


for name, fn in [("isarcasm", isarcasm), ("sarc", sarc), ("talkdown", talkdown), ("mentalmanip", mentalmanip),
                 ("casino", casino), ("esconv", esconv), ("empathetic_dialogues", empathetic_dialogues),
                 ("semeval18", semeval18), ("measuring_hate", hate_measure), ("politeness", politeness), ("swda", swda),
                 ("dreaddit", dreaddit), ("dailydialog", dailydialog), ("trac2", trac2), ("hindi_sarcasm", hindi_sarcasm),
                 ("hinglish_sarcasm", hinglish_sarcasm), ("ben_sarc", ben_sarc), ("bengali_humor", bengali_humor),
                 ("bad_bengali", bad_bengali), ("bangla_fb", bangla_fb), ("emonoba", emonoba), ("bengali_80k", bengali_80k),
                 ("whatsapp", whatsapp)]:
    run(name, fn)

random.shuffle(rows)
with gzip.open(os.path.join(OUT, "tones.jsonl.gz"), "wt", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
pos = collections.Counter(t for r in rows for t, v in r["tones"].items() if v >= 1.0)
summary = {"rows": len(rows), "per_source": dict(stats), "per_language": dict(collections.Counter(r["l"] for r in rows)),
           "tone_positives": dict(pos.most_common()), "tones_without_positives": sorted(TONES - set(pos)),
           "styles": dict(collections.Counter(r["style"] for r in rows if r["style"])),
           "samples": [(r["s"], r["t"][:100], r["tones"], r["style"]) for r in rows[:40]]}
json.dump(summary, open(os.path.join(OUT, "stats.json"), "w"), ensure_ascii=False, indent=1)
log(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False, indent=1))
