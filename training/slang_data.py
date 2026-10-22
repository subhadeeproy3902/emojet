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

SMOKE = __SMOKE__
TAXONOMY_B64 = "__TAXONOMY__"
CHATLEX_B64 = "__CHATLEX__"            # training/chatlex.py
CHAT_LEXICON_B64 = "__CHAT_LEXICON__"    # training/data/chat_lexicon.json
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
