"""Emojet slang + swearing data (runs on Kaggle, CPU, internet on). Round 2.

1. Lexicons of slang and swear words (EN, Hindi, Hinglish, Bengali, Banglish) from open lists, plus a hand list
   of everyday Hinglish/Banglish chat slang that no open list covers.
2. REAL messages that use those words: mined from the emojet-data real messages (attached kernel output) and from
   real slang/abuse datasets. These become targets for the tone/emoji labeller (annotate.py, round 2).
3. The human labels those datasets already carry (vulgar, troll, threat, insult, hate type) mapped to the taxonomy.

Output /kaggle/working/:
  slang/lexicon.json        {variety: [{"term", "src", "severity"?, "meaning"?}]}
  slang/targets.jsonl.gz    {"t", "l", "s": "slang-<lang>", "e": [emoji ids typed], "term"}   -> annotate round 2
  tones/tones.jsonl.gz      same format as tone_data.py (human labels), read by the training notebook
  slang/stats.json

Built by training/build_slang.py (fills in TAXONOMY_B64 and SMOKE).
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
import subprocess
import sys
import time
import urllib.request

SMOKE = True
TAXONOMY_B64 = "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"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "emoji==2.14.1"], check=False)
import emoji  # noqa: E402
from datasets import load_dataset  # noqa: E402

T0 = time.time()
random.seed(17)
csv.field_size_limit(10_000_000)
for d in ("/kaggle/working/slang", "/kaggle/working/tones"):
    os.makedirs(d, exist_ok=True)
TAX = json.loads(gzip.decompress(base64.b64decode(TAXONOMY_B64)).decode())
TONES, STYLES, EMOJIS = set(TAX["tones"]), set(TAX["styles"]["options"]), TAX["emojis"]
SKIN = {chr(c) for c in range(0x1F3FB, 0x1F400)}
E_INDEX = {}
for i, e in enumerate(EMOJIS):
    E_INDEX.setdefault("".join(ch for ch in e if ch not in SKIN and ch != "️"), i)
LIMIT = 2000 if SMOKE else None
GH = "https://raw.githubusercontent.com"
KG = "https://www.kaggle.com/api/v1/datasets/download/"


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "emojet"}), timeout=300).read()


def kaggle_file(ref, path):
    """One file of a public Kaggle dataset (the download is a zip when the file is compressed)."""
    raw = fetch(f"{KG}{ref}/{urllib.parse.quote(path)}")
    if raw[:2] == b"PK":
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(z.namelist()[0])
    return raw


import urllib.parse  # noqa: E402


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


def run(name, fn):
    t = time.time()
    try:
        fn()
    except Exception as exc:
        log(f"!! {name} failed: {exc!r}")
    log(f"{name} done ({time.time() - t:.0f} s)")


# ---------------------------------------------------------------- 1. lexicons
lex = collections.defaultdict(dict)        # variety -> term -> info


def term(variety, t, src, **info):
    t = str(t or "").strip().lower()
    if 2 <= len(t) <= 40 and t not in lex[variety]:
        lex[variety][t] = {"src": src, **{k: v for k, v in info.items() if v not in (None, "")}}


# everyday chat slang that open lists miss (mild, mostly not swearing); used only to FIND real messages
HAND = {
    "banglish": """ekdom ekdum haga hagu hagbo hagte mut mutbo pad padchi matha kharap mathakharap pagol pagla pagli fatafati
        jhakkas jhakas darun osthir oshthir cholbe na jah dhur dhurr bokachoda boka choda shala sala bal baal chudir khanki
        magi tor mayer kutta harami hala halar put chagol gadha ganja dhon fokinni fokir chapri bhab nibi pera nai chill
        mama vai bhai bro dosto jhamela pera khaisi khaise kheyechi khabo ghumabo ghum pacche haga pacche mut pacche
        boro lok pocha jinish faltu bekar baje jhal moja moja lagse moja pailam lol hasi pai hasi ashche""".split(),
    "hinglish": """ekdum jhakaas jhakas bindaas bakwas bakwaas faltu jugaad jugad scene kadak pataka ghanta bas kar chill maar
        mast fadu jhakkas bhaukal bawal bawaal chamak bhai yaar yr tapori lafda locha panga chindi chamcha kamina kameena
        ullu pagal paagal bewakoof bevakoof gadha saala sala chutiya bc mc bsdk bkl bhenchod madarchod lodu lode lund
        gaand gand harami kutta kutte kamine hagna hag raha potty susu tatti dimag kharab dimaag kharab bhaad mein jao
        jaa na chal hat oye arre yaar kya scene hai fattu dhinchak senti filmy""".split(),
}


def lexicons():
    for t in fetch(f"{GH}/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en").decode().splitlines():
        term("english", t, "ldnoobw")
    for t in fetch(f"{GH}/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/hi").decode().splitlines():
        term("hinglish", t, "ldnoobw")
    for row in csv.reader(io.StringIO(fetch(f"{GH}/pmathur5k10/Hinglish-Offensive-Text-Classification/master/Hinglish_Profanity_List.csv").decode("latin-1"))):
        if row:
            term("hinglish", row[0], "mathur", meaning=row[1] if len(row) > 1 else None, severity=row[2].strip() if len(row) > 2 else None)
    for fname, variety in (("hi-latin", "hinglish"), ("hi-deva", "hindi"), ("en", "english")):
        try:
            data = json.loads(fetch(f"{GH}/mathpalnaman/hinglish-profanity/main/src/wordlists/{fname}.json"))
            for r in (data if isinstance(data, list) else data.get("words", [])):
                term(variety, r.get("term") if isinstance(r, dict) else r, "mathpalnaman",
                     severity=r.get("severity") if isinstance(r, dict) else None)
        except Exception as exc:
            log("  mathpalnaman", fname, exc)
    js = fetch(f"{GH}/surajJha/profanity-hindi/master/data/hindi-bad-words.js").decode("utf-8", "replace")
    for t in re.findall(r"""['"]([^'"]{2,40})['"]\s*:""", js):
        term("hinglish", t, "surajJha")
    for r in csv.DictReader(io.StringIO(kaggle_file("ikkiocean/english-and-hindi-swear-words-dataset", "bad_words.csv").decode("utf-8", "replace"))):
        v = {"english": "english"}.get((r.get("language") or "").lower(), None)
        script = (r.get("script") or "").lower()
        if (r.get("language") or "").lower().startswith("hi"):
            v = "hindi" if "deva" in script else "hinglish"
        if v and not re.search(r"[0-9@$]", r.get("swear_word") or ""):      # skip leetspeak variants
            term(v, r.get("swear_word"), "ikkiocean")
    raw = kaggle_file("shadow4ever/top-112-hindi-swear-words", "hindi top 150 swear words .csv").decode("utf-8", "replace")
    for row in csv.reader(io.StringIO(raw)):
        for cell in row:
            term("hindi" if re.search("[ऀ-ॿ]", cell) else "hinglish", cell, "shadow4ever")
    for r in csv.DictReader(io.StringIO(kaggle_file("newton62/slang-dataset-for-indian-reg-lang", "slang_dataset.csv").decode("utf-8", "replace"))):
        for k, v in r.items():
            if k and "hindi" in k.lower():
                term("hinglish", v, "newton62", meaning=next(iter(r.values())))
    for line in fetch(f"{GH}/amirulsizan/deshi-slang/main/data/profanity_dataset.jsonl").decode("utf-8", "replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        term("bengali", r.get("text"), "deshi-slang", severity=r.get("severity_score"))
        for rom in (r.get("romanized") if isinstance(r.get("romanized"), list) else [r.get("romanized")]):
            term("banglish", rom, "deshi-slang", severity=r.get("severity_score"))
    for line in fetch(f"{GH}/Sigmakib2/Nirmol/main/datasets/Nirmol-v1-dataset.csv").decode("utf-8", "replace").splitlines()[1:]:
        term("bengali", line.split(",")[0].strip('" '), "nirmol")
    for fname, variety in (("bengali", "bengali"), ("hindi", "hindi")):
        js = fetch(f"{GH}/kuntalojha/slangs/main/languages/{fname}.js").decode("utf-8", "replace")
        for t in re.findall(r"""['"]([^'"]{2,40})['"]""", js):
            term("hindi" if re.search("[ऀ-ॿ]", t) else ("bengali" if re.search("[ঀ-৿]", t) else
                 ("banglish" if variety == "bengali" else "hinglish")), t, "kuntalojha")
    for variety, words in HAND.items():
        for t in words:
            term(variety, t, "emojet-hand")
    for variety, words in (("banglish", ["matha kharap", "haga pacche", "mut pacche", "ghum pacche", "boro lok", "moja pailam"]),
                           ("hinglish", ["bas kar", "chill maar", "dimag kharab", "bhaad mein jao", "kya scene hai", "hag raha"])):
        for t in words:
            term(variety, t, "emojet-hand")


run("lexicons", lexicons)
json.dump({v: [{"term": t, **i} for t, i in d.items()] for v, d in lex.items()},
          open("/kaggle/working/slang/lexicon.json", "w"), ensure_ascii=False, indent=0)
log("lexicon sizes:", {v: len(d) for v, d in lex.items()})

# ---------------------------------------------------------------- 2. human-labelled real slang/abuse messages
tone_rows = []


def tone_row(src, text, lang, tones=None, style=None):
    text = clean(text)
    tones = {k: float(v) for k, v in (tones or {}).items()}
    assert all(k in TONES for k in tones) and (style is None or style in STYLES), (src, tones, style)
    if 2 <= len(text) <= 1000 and (tones or style):
        tone_rows.append({"t": text, "r": None, "l": lang, "s": src, "tones": tones, "style": style, "e": []})


labelled_texts = []                        # (text, lang, src, emojis) of real messages from these sets


def keep(text, lang, src):
    if isinstance(text, str) and text.strip():
        labelled_texts.append((text, lang, src))


JK = {"Sexual": ["rude"], "Slang": ["rude"], "Appearance": ["derisive", "rude"], "Religious": ["aggressive"],
      "Racial": ["aggressive", "derisive"], "Others": ["rude"]}


def jkkamal():
    rd = csv.DictReader(io.StringIO(kaggle_file("jkkamal/banglish-hate-spech", "Banglish Hate Speech Dataset.csv").decode("utf-8", "replace")))
    for k, r in enumerate(rd):
        if LIMIT and k >= LIMIT: break
        typ = (r.get("Type") or "").strip()
        if typ in JK:
            tone_row("banglish_hate", r["Comment"], "bn", {t: 1.0 for t in JK[typ]}, "aggressive")
        elif typ == "Not Hate":
            tone_row("banglish_hate", r["Comment"], "bn", {"rude": 0.0, "aggressive": 0.0})
        keep(r.get("Comment"), "bn", "banglish_hate")


def bn_multilabel():
    for k, r in enumerate(load_dataset("sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data", split="train")):
        if LIMIT and k >= LIMIT: break
        tones = {}
        if r.get("Insult"): tones["rude"] = 1.0
        if r.get("Violence"): tones["aggressive"] = 1.0
        if r.get("Sexual"): tones["rude"] = 1.0
        if str(r.get("Hate/NoHate", "")).lower() in ("0", "nohate", "no hate", "false"):
            tones = {"rude": 0.0, "aggressive": 0.0}
        tone_row("bn_multilabel_hate", r.get("Comment"), "bn", tones)
        keep(r.get("Comment"), "bn", "bn_multilabel_hate")


def bn_toxic():
    rd = csv.DictReader(io.StringIO(kaggle_file("tanveerbelaliut/multi-labeled-bengali-toxic-comments", "Multi_labeled_toxic_comments.csv").decode("utf-8", "replace")))
    m = {"vulgar": ["rude"], "hate": ["aggressive"], "threat": ["aggressive"], "troll": ["derisive"], "Insult": ["rude", "derisive"]}
    for k, r in enumerate(rd):
        if LIMIT and k >= LIMIT: break
        tones = {t: 1.0 for c, ts in m.items() if str(r.get(c, "0")).strip() == "1" for t in ts}
        if not tones:
            tones = {"rude": 0.0, "aggressive": 0.0, "derisive": 0.0}
        tone_row("bn_toxic", r.get("text"), "bn", tones)
        keep(r.get("text"), "bn", "bn_toxic")


def hinglish_hate():
    for k, r in enumerate(load_dataset("pankajbiswas6/prism-hinglish-hate-speech", split="train")):
        if LIMIT and k >= LIMIT: break
        lang = {"hinglish": "hi", "hindi": "hi", "english": "en"}.get(str(r.get("lang", "")).lower(), "hi")
        hate = str(r.get("label")) in ("1", "hate", "True")
        tone_row("hinglish_hate", r.get("text"), lang, {"aggressive": 1.0, "rude": 1.0} if hate else {"aggressive": 0.0})
        keep(r.get("text"), lang, "hinglish_hate")


def hasoc():
    for k, r in enumerate(load_dataset("nikitadesai/hasoc", split="train")):
        if LIMIT and k >= LIMIT: break
        text = r.get("rawtweets") or r.get("text")
        cls = str(r.get("binary class") or r.get("label") or "")
        tone_row("hasoc_hi", text, "hi", {"aggressive": 1.0, "rude": 1.0} if cls == "HOF" else ({"aggressive": 0.0} if cls == "NOT" else {}))
        keep(text, "hi", "hasoc_hi")


for name, fn in [("jkkamal", jkkamal), ("bn_multilabel", bn_multilabel), ("bn_toxic", bn_toxic),
                 ("hinglish_hate", hinglish_hate), ("hasoc", hasoc)]:
    run(name, fn)

# ---------------------------------------------------------------- 3. find real messages that use the slang
LANG_OF = {"banglish": "bn", "bengali": "bn", "hinglish": "hi", "hindi": "hi", "english": "en"}
roman_terms = {t: v for v in ("banglish", "hinglish") for t in lex[v]}
script_terms = {t: v for v in ("bengali", "hindi") for t in lex[v] if len(t) >= 3}
WORD = re.compile(r"[a-z]+")
PER_TERM = 60 if SMOKE else 400
per_term = collections.Counter()
targets, seen = [], set()


SINGLE = {t for t in roman_terms if " " not in t}
MULTI = [t for t in roman_terms if " " in t]
SCRIPT_RE = re.compile("|".join(sorted(map(re.escape, script_terms), key=len, reverse=True))) if script_terms else None


def match(text):
    """The rarest-so-far slang term in the message (so common words like 'bhai' don't crowd out rare ones)."""
    low = text.lower()
    words = WORD.findall(low)
    cands = [(t, roman_terms[t]) for t in set(words) & SINGLE]
    if MULTI:
        joined = " " + " ".join(words) + " "
        cands += [(t, roman_terms[t]) for t in MULTI if (" " + t + " ") in joined]
    if SCRIPT_RE:
        cands += [(t, script_terms[t]) for t in set(SCRIPT_RE.findall(low))]
    cands = [c for c in cands if per_term[c[0]] < PER_TERM]
    return min(cands, key=lambda c: per_term[c[0]]) if cands else None


def offer(raw_text, lang, src, typed=None):
    text = clean(raw_text)
    if not 3 <= len(text) <= 500:
        return
    m = match(text)
    if not m:
        return
    t, variety = m
    key = text.lower()
    if key in seen:
        return
    seen.add(key)
    per_term[t] += 1
    targets.append({"t": text, "l": LANG_OF.get(variety, lang), "s": "slang-" + variety, "e": typed or emoji_labels(raw_text), "term": t})


def mine():
    for text, lang, src in labelled_texts:
        offer(text, lang, src)
    log("  from slang/abuse sets:", len(targets))
    for path in sorted(glob.glob("/kaggle/input/**/real/train.jsonl.gz", recursive=True) +
                       glob.glob("/kaggle/input/**/real/plain.jsonl.gz", recursive=True)):
        for k, line in enumerate(gzip.open(path, "rt", encoding="utf-8")):
            if LIMIT and k >= LIMIT * 20: break
            r = json.loads(line)
            if r["l"] in ("bn", "hi"):
                offer(r["t"], r["l"], r.get("s"), typed=r.get("e"))
        log("  after", os.path.basename(path), len(targets))


def urban_dictionary():
    ds = load_dataset("georgiyozhegov/urbandictionary", split="train")
    rows = sorted((r for r in ds if r.get("example") and (r.get("score") or 0) > 0), key=lambda r: -(r.get("score") or 0))
    n = 0
    for r in rows[: (300 if SMOKE else 12_000)]:
        text = clean(r["example"])
        if 3 <= len(text) <= 400 and text.lower() not in seen:
            seen.add(text.lower())
            targets.append({"t": text, "l": "en", "s": "slang-english", "e": [], "term": str(r.get("word", ""))[:40]})
            n += 1
    log("  urban dictionary examples:", n)


run("mine", mine)
run("urban_dictionary", urban_dictionary)

random.shuffle(targets)
with gzip.open("/kaggle/working/slang/targets.jsonl.gz", "wt", encoding="utf-8") as fh:
    for r in targets:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with gzip.open("/kaggle/working/tones/tones.jsonl.gz", "wt", encoding="utf-8") as fh:
    for r in tone_rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
stats = {"lexicon": {v: len(d) for v, d in lex.items()}, "targets": len(targets),
         "targets_per_variety": dict(collections.Counter(r["s"] for r in targets)),
         "tone_rows": len(tone_rows), "tone_rows_per_source": dict(collections.Counter(r["s"] for r in tone_rows)),
         "top_terms": per_term.most_common(60),
         "owner_terms": {t: per_term.get(t, 0) for t in ["haga", "hagu", "haga pacche", "ekdom", "matha kharap", "pagol", "bokachoda",
                                                         "ekdum", "jhakaas", "bc", "chutiya", "bsdk"]},
         "samples": [(r["s"], r["term"], r["t"][:100]) for r in targets[:40]]}
json.dump(stats, open("/kaggle/working/slang/stats.json", "w"), ensure_ascii=False, indent=1)
log(json.dumps({k: v for k, v in stats.items() if k != "samples"}, ensure_ascii=False, indent=1))
for s in stats["samples"][:25]:
    print(s)
