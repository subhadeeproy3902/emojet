"""Emojet slang + swearing + everyday chat words (runs on Kaggle, CPU, internet on). Round 3.

1. Lexicons of slang and swear words (EN, Hindi, Hinglish, Bengali, Banglish) from open lists, plus a hand list
   of everyday Hinglish/Banglish chat slang that no open list covers.
2. REAL messages that use those words: mined from the emojet-data real messages (attached kernel output) and from
   real slang/abuse datasets. These become targets for the tone/emoji labeller (annotate.py, round 2).
3. The human labels those datasets already carry (vulgar, troll, threat, insult, hate type) mapped to the taxonomy.

Output /kaggle/working/:
  slang/lexicon.json        {variety: [{"term", "src", "severity"?, "meaning"?}]}
  slang/targets.jsonl.gz    {"t", "l", "s": "slang-<variety>", "e": [emoji ids typed], "term", "kind", "curated", "notes"}  -> annotate
  slang/lexicon_used.json   curated chat lexicon + the external terms that survived the common-word filter
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

SMOKE = False
TAXONOMY_B64 = "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"
CHATLEX_B64 = "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"            # training/chatlex.py
CHAT_LEXICON_B64 = "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"    # training/data/chat_lexicon.json
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

# ---------------------------------------------------------------- 3. find real messages that use the words
# Round 3: one matcher for everything (training/chatlex.py, inlined below): whole words, spelling folding, and the
# language of the MESSAGE, not of the list a word came from. Round 2 matched raw substrings ("haga" inside
# "bhagao", "nti" inside any word): half of its 67k "slang" messages had no slang in them.
_ns = {}
exec(gzip.decompress(base64.b64decode(CHATLEX_B64)).decode("utf-8"), _ns)
Lexicon = _ns["Lexicon"]
CURATED = json.loads(gzip.decompress(base64.b64decode(CHAT_LEXICON_B64)).decode("utf-8"))["entries"]
LANG_OF = {"banglish": "bn", "bengali": "bn", "hinglish": "hi", "hindi": "hi", "english": "en"}
curated_terms = {t.lower() for e in CURATED for t in e["roman"] + e["native"]}
external = []
for variety, d in lex.items():
    for t, info in d.items():
        if t in curated_terms or re.search(r"[0-9@$*#!_.]", t):                  # leetspeak: rare in real chat
            continue
        if info["src"] == "emojet-hand" or (info["src"] == "deshi-slang" and str(info.get("severity")) == "1"):
            continue                       # round 2's hand list ("kya", "tor") -> the curated lexicon; severity 1 = ordinary words
        if t.isascii() and (len(re.sub(r"[^a-z]", "", t)) < 3 or len(t.split()) > 4):
            continue
        native = not t.isascii()
        external.append({"lang": LANG_OF[variety], "kind": "swear", "roman": [] if native else [t], "native": [t] if native else [],
                         "gloss": (info.get("meaning") or "insult, swear or crude slang") + " (from a profanity list; can also be an "
                                  "ordinary word, judge from context)", "emojis": "", "inflect": False, "src": info["src"]})

real_files = sorted(glob.glob("/kaggle/input/**/real/train.jsonl.gz", recursive=True) +
                    glob.glob("/kaggle/input/**/real/plain.jsonl.gz", recursive=True))
def script_of(text):
    return "native" if re.search("[\u0900-\u09ff]", text) else "roman"


sample, n_pop = [], collections.Counter()
for path in real_files:
    for k, line in enumerate(gzip.open(path, "rt", encoding="utf-8")):
        if k % 25 == 0:
            r = json.loads(line)
            sample.append(r)
            n_pop[(Lexicon.lang_of(r["t"], r["l"]), script_of(r["t"]))] += 1
        if LIMIT and k >= LIMIT * 20: break
# A listed word in more than 0.1% of ordinary messages of its own language AND script is just a word (मारना = to hit).
# Per script: Hindi data is mostly Devanagari, which hid how common romanized "kya" is in Hinglish.
L_ext = Lexicon(external)
df = collections.Counter(k for r in sample for k in {k for k, _ in L_ext.find(r["t"], r["l"])})
def pop(e):
    sc = "native" if e["native"] else "roman"
    return sum(n for (l, s_), n in n_pop.items() if s_ == sc and (e["lang"] == "en" or l == e["lang"]))
TRUSTED = {"ldnoobw", "mathur", "shadow4ever"}   # curated profanity lists: their words are swears even when common
common = {k for k, n in df.items() if external[k]["src"] not in TRUSTED and n / max(1, pop(external[k])) > 0.001}
log(f"external terms: {len(external):,}, dropped as common words: {len(common)}:",
    [external[k]["roman"][:1] + external[k]["native"][:1] for k in sorted(common, key=lambda k: -df[k])[:60]])
ALL = CURATED + [e for k, e in enumerate(external) if k not in common]
L = Lexicon(ALL)
PER_ENTRY = {"topic": 700, "reaction": 300, "swear": 150, "address": 0}
per = collections.Counter()
targets, seen = [], set()


def offer(raw_text, lang, src, typed=None):
    text = clean(raw_text)
    if not 3 <= len(text) <= 500 or text.lower() in seen:
        return
    hits = [(k, t) for k, t in L.find(text, lang) if per[k] < PER_ENTRY[ALL[k]["kind"]] // (10 if SMOKE else 1)]
    if not hits:
        return
    k, t = min(hits, key=lambda h: (per[h[0]], h[0] >= len(CURATED)))     # rarest so far; curated wins ties
    seen.add(text.lower())
    per[k] += 1
    msg_lang = Lexicon.lang_of(text, lang)
    roman = not re.search("[ऀ-৿]", text)
    variety = {"bn": "banglish" if roman else "bengali", "hi": "hinglish" if roman else "hindi"}.get(msg_lang, "english")
    targets.append({"t": text, "l": msg_lang, "s": "slang-" + variety, "e": typed if typed is not None else emoji_labels(raw_text),
                    "term": t, "kind": ALL[k]["kind"], "curated": k < len(CURATED), "notes": L.notes(text, lang), "src": src})


def mine():
    for text, lang, src in labelled_texts:
        offer(text, lang, src)
    log("  from slang/abuse sets:", len(targets))
    for path in real_files:
        for k, line in enumerate(gzip.open(path, "rt", encoding="utf-8")):
            if LIMIT and k >= LIMIT * 20: break
            r = json.loads(line)
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
            targets.append({"t": text, "l": "en", "s": "slang-english", "e": [], "term": str(r.get("word", ""))[:40], "kind": "swear",
                            "curated": False, "notes": [f"{str(r.get('word', ''))[:40]} = {clean(r.get('definition', ''))[:120]}"], "src": "urbandictionary"})
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
stats = {"lexicon": {v: len(d) for v, d in lex.items()}, "external_kept": len(ALL) - len(CURATED), "external_common": len(common),
         "targets": len(targets), "targets_per_variety": dict(collections.Counter(r["s"] for r in targets)),
         "targets_per_kind": dict(collections.Counter(r["kind"] for r in targets)),
         "tone_rows": len(tone_rows), "tone_rows_per_source": dict(collections.Counter(r["s"] for r in tone_rows)),
         "curated": {f"{e['lang']} {(e['roman'] or e['native'])[0]}": per[k] for k, e in enumerate(CURATED)},
         "top_external": [((ALL[k]["roman"] or ALL[k]["native"])[0], n) for k, n in per.most_common() if k >= len(CURATED)][:60],
         "samples": [(r["s"], r["term"], r["notes"][:1], r["t"][:100]) for r in targets[:60]]}
json.dump(stats, open("/kaggle/working/slang/stats.json", "w"), ensure_ascii=False, indent=1)
json.dump(ALL, open("/kaggle/working/slang/lexicon_used.json", "w"), ensure_ascii=False)
log(json.dumps({k: v for k, v in stats.items() if k != "samples"}, ensure_ascii=False, indent=1))
for s_ in stats["samples"][:40]:
    print(s_)
# every target must really contain its word (round 2's bug); fail loudly if the matcher regresses
assert all(any(t == r["term"] for _, t in L.find(r["t"], r["l"])) for r in targets if r["src"] != "urbandictionary"), "target without its word"
