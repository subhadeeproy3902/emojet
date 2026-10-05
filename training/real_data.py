"""Emojet real-data builder (runs on Kaggle, CPU, internet on).

Real messages people wrote, in English, Hindi (Devanagari + romanized) and Bengali (script +
romanized Banglish). The emojis a person typed are the labels (distant supervision): they are
removed from the text and become what the model has to predict.

Output in /kaggle/working/real/:
  train.jsonl.gz   {"t": text, "e": [emoji ids, most used first], "l": "en"|"hi"|"bn", "s": source}
  test.jsonl.gz    2,000 held-out messages per language (never used for training)
  stats.json       counts per source / language, emoji coverage, samples

Built by training/build_data.py, which fills in CATALOG_B64 and SMOKE.
"""
import base64
import collections
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

SMOKE = __SMOKE__
CATALOG_B64 = "__CATALOG__"

for _pkg in ("emoji==2.14.1", "ijson", "fasttext-predict"):      # one at a time: a failure must not block the others
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)
import emoji  # noqa: E402
import ijson  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

T0 = time.time()
random.seed(13)
OUT = "/kaggle/working/real"
DL = "/tmp/dl"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DL, exist_ok=True)


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


# ---------------------------------------------------------------- labels
EMOJIS = json.loads(gzip.decompress(base64.b64decode(CATALOG_B64)).decode())
SKIN = {chr(c) for c in range(0x1F3FB, 0x1F400)}


def norm(e):
    return "".join(ch for ch in e if ch not in SKIN and ch != "️")


E_INDEX = {}
for i, e in enumerate(EMOJIS):
    E_INDEX.setdefault(norm(e), i)
# fast pre-check: any char in the usual emoji blocks
HAS_EMOJI = re.compile("[©®‼-㊙\U0001F000-\U0001FAFF]")
NAME = re.compile(r":[a-z0-9_&'.\-]+:")


def labels(text):
    """Distinct catalog emojis, most repeated first, then first seen."""
    cnt, first = collections.Counter(), {}
    for k, m in enumerate(emoji.emoji_list(text)):
        i = E_INDEX.get(norm(m["emoji"]))
        if i is None:
            continue
        cnt[i] += 1
        first.setdefault(i, k)
    return sorted(cnt, key=lambda i: (-cnt[i], first[i]))[:6]


# ---------------------------------------------------------------- text cleaning
URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|in|bd|ly|me|io|net|org)/\S*)", re.I)
MENTION = re.compile(r"(^|\s)@\w+")
RT = re.compile(r"^RT\s+@\w+:?\s*")
SPACES = re.compile(r"\s+")
LEFTOVER = re.compile("[‍️⃣\U0001F3FB-\U0001F3FF]")
SPAM = re.compile(r"airdrop|giveaway|follow\s*(me|back|for\s*follow)|subscribe|sub4sub|t\.me/|telegram|whatsapp\s*\+?\d"
                  r"|crypto|\bnft\b|bitcoin|\$[A-Z]{3,}|promo\s*code|link\s*in\s*bio|check\s*out\s*my|dm\s*(me|for)"
                  r"|earn\s*money|join\s*now|free\s*followers|discount\s*code|\bcashback\b|#ad\b", re.I)


def clean(text):
    text = RT.sub("", text)
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = emoji.replace_emoji(text, " ")
    text = LEFTOVER.sub("", text).replace("#", "")
    return SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------- language
BN_CHARS = re.compile("[ঀ-৿]")
HI_CHARS = re.compile("[ऀ-ॿ]")
LATIN = re.compile("[A-Za-z]")
LETTERS = re.compile(r"[^\W\d_]")
WORD = re.compile(r"[a-z']+")
# frequent romanized words (lower case) that English almost never uses
ROMAN_HI = set("""hai hain nahi nhi nahin kya kyu kyun kyon bhai yaar yr mera meri mere tera teri tere accha acha achha bahut
bohot bhot sab kuch aur karo kar raha rahi rahe gaya gayi hoga hogi tum aap mai mujhe tujhe humko sahi ekdum matlab kaise kaisa
kab abhi bas wala wali waale ji haan nai bilkul sach pyaar pyar dost ghar paisa log logo yeh woh jo toh tha thi the kaha kahan
chal chalo dekho dekh suno bolo bol mat lekin magar phir fir kab kitna kitne apna apni jaldi bekar mast bakwas bc mc bsdk saala
chutiya bhaiya didi beta arre arey oye hum unka uska iska inka diya liya kiya karna hona jana""".split())
ROMAN_BN = set("""ami tumi apni amra tomra ki keno kore korbo korchi korche korcho korlam bhalo valo achi acho achen ache ase
asi kotha bolchi bolo boli khub onek ekdom haga hagu pacche pache hobe hoye hoe geche gechi chilo chhilo theke amar tomar tor
mone kintu ekta kichu kisu darun osadharon oshadharon vai bhaiya dada didi jabo khabo jacchi jachchi korte dekhe shob sob matha
thik pore ajke ajk kemon nai nei pagol shala sala baje bal bokachoda khanki magi chudi tomake amake take ekhon ekhane okhane
hocche hochhe hoyeche hoyche lagche lagse lage dekho dekhi bujhlam bujhi jani jano parbo pari parina na kno abar tai tahole
sathe shathe kothay kobe kivabe kibhabe shundor sundor valobashi bhalobashi bhalobasha valobasa koto boro choto kaj kaaj
porashona khete ghum ghumabo shuvo subho dhonnobad""".split())
ROMAN_SHARED = {"bhai", "na", "nai", "dada", "didi", "bhaiya"}
ROMAN_HI -= {"the", "mere", "main", "mat", "bas", "log", "hum", "ho", "jo", "kar", "bol", "fir", "mai", "sab", "na", "ji",
             "diya", "tha", "aur", "wala", "yeh", "chal", "mast"}
ROMAN_BN -= {"take", "ache", "ase", "asi", "bal", "pari", "jani", "tai", "lage", "boro", "mone", "magi", "na", "kaj", "sala",
             "abar", "shob", "sob", "ki", "koto", "baje", "dekhi", "pore"}
EN_STOP = set("""the and you that this for are with have not but what all was your just can get like love when one out
so about how will they from there more good know time she him his her our them would been their who did really
it's i'm don't can't im dont thank thanks lol omg please today day best happy new great""".split())

_ft = None
try:
    import fasttext
    path = os.path.join(DL, "lid.176.ftz")
    subprocess.run(["wget", "-q", "-O", path, "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"], check=True)
    _ft = fasttext.load_model(path)
except Exception as exc:  # fall back to stopwords
    log("fastText unavailable, using stopwords:", exc)


def ft_lang(text):
    try:
        labs, probs = _ft.predict(text.replace("\n", " "), k=1)
        return labs[0].replace("__label__", ""), float(probs[0])
    except Exception:
        return None, 0.0


def language(text, hint=None):
    """'en' | 'hi' | 'bn' | None (other language / unclear)."""
    letters = LETTERS.findall(text)
    if len(letters) < 2:
        return None
    n = len(letters)
    bn, hi, la = len(BN_CHARS.findall(text)), len(HI_CHARS.findall(text)), len(LATIN.findall(text))
    if bn / n >= 0.3:
        return "bn"
    if hi / n >= 0.3:
        return "hi"
    if la / n < 0.7:
        return None                                    # Arabic, Cyrillic, CJK, ...
    words = WORD.findall(text.lower())
    rh = sum(w in ROMAN_HI and w not in ROMAN_SHARED for w in words)
    rb = sum(w in ROMAN_BN and w not in ROMAN_SHARED for w in words)
    # romanized Hindi/Bengali: a real share of its own words, not one lookalike in an English sentence
    def roman(k):
        return k >= 1 and (len(words) <= 4 or k >= 3 or k / max(1, len(words)) >= 0.15)
    if roman(rb) and rb >= rh:
        return "bn"
    if roman(rh):
        return "hi"
    if hint in ("bn", "hi") and (rb or rh):           # a Bengali/Hindi source with a romanized word
        return "bn" if rb >= rh else "hi"
    if _ft is not None:
        lab, p = ft_lang(text)
        if lab == "en" and p >= 0.5:
            return "en"
        if lab in ("hi", "bn", "mr", "ne") and hint in ("bn", "hi"):
            return hint
        if hint in ("bn", "hi") and lab != "en":
            return hint if (rb or rh) else None
        return None
    return "en" if any(w in EN_STOP for w in words) else None


# ---------------------------------------------------------------- collection
QUOTA = {"en": 1_600_000, "hi": 700_000, "bn": 10_000_000} if not SMOKE else {"en": 6000, "hi": 4000, "bn": 4000}
CAP = {"en": 30_000, "hi": 40_000, "bn": 25_000} if not SMOKE else {"en": 300, "hi": 300, "bn": 300}  # per first emoji
kept = {l: [] for l in QUOTA}
first_count = {l: collections.Counter() for l in QUOTA}
seen = set()
stats = collections.defaultdict(collections.Counter)


def offer(raw, src, hint=None, names=False):
    """One real message -> maybe one example."""
    st = stats[src]
    st["read"] += 1
    if not isinstance(raw, str) or not HAS_EMOJI.search(raw) and not (names and ":" in raw):
        return
    if names:
        raw = emoji.emojize(raw, language="alias")
    if raw.startswith("RT @"):
        st["retweet"] += 1
        return
    ids = labels(raw)
    if not ids:
        return
    st["with_emoji"] += 1
    text = clean(raw)
    if len(text) < 2 or len(text) > 1200:
        st["length"] += 1
        return
    if SPAM.search(text) or text.count("#") > 3:
        st["spam"] += 1
        return
    lang = language(text, hint)
    if lang is None:
        st["other_lang"] += 1
        return
    key = hashlib.blake2b((re.sub(r"\W+", "", text.lower()) + "|" + ",".join(map(str, ids))).encode(), digest_size=8).digest()
    if key in seen:
        st["duplicate"] += 1
        return
    seen.add(key)
    if len(kept[lang]) >= QUOTA[lang]:
        st["quota"] += 1
        return
    if first_count[lang][ids[0]] >= CAP[lang]:      # 😂 / ❤️ / 🙏 would otherwise drown everything else
        st["capped"] += 1
        return
    first_count[lang][ids[0]] += 1
    kept[lang].append({"t": text[:1200], "e": ids, "l": lang, "s": src})
    st["kept"] += 1
    st["kept_" + lang] += 1


def full(lang):
    return len(kept[lang]) >= QUOTA[lang]


def download(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=DL)


def parquet_texts(repo, files, column):
    for f in files:
        path = download(repo, f)
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=[column]).column(0).to_pylist():
                yield t
        os.remove(path)


def jsonl_texts(path, field="text"):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                yield json.loads(line).get(field)
            except ValueError:
                continue


def run(name, fn):
    t = time.time()
    try:
        fn()
    except Exception as exc:                         # one broken source must not kill the run
        log(f"!! {name} failed: {exc!r}")
    log(f"{name}: {dict(stats[name])}  ({time.time() - t:.0f} s)  kept so far",
        {l: len(v) for l, v in kept.items()})


LIMIT = 20_000 if SMOKE else None                     # rows per source in smoke mode


def take(it):
    for k, x in enumerate(it):
        if LIMIT and k >= LIMIT:
            return
        yield x


# ---- Bengali / Banglish (scarcest: all of it, first)
def banglishrev():
    path = download("BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset", "reviews v1.json")
    with open(path, "rb") as fh:
        for t in take(ijson.items(fh, "item.Reviews.item.Review Content")):
            offer(t, "banglishrev", "bn")
    os.remove(path)


def bn_youtube():
    for t in take(parquet_texts("irfanhossainsust/bangla-youtube-comments-sentiment",
                                ["data/train.parquet", "data/validation.parquet", "data/test.parquet"], "text")):
        offer(t, "bn_youtube", "bn")


def coke_studio():
    for t in take(parquet_texts("ragibhasan/coke_studio_bangla_sentiment", ["data/full/train-00000-of-00001.parquet"], "text_raw")):
        offer(t, "coke_studio", "bn")


def banglish_dep_yt():
    import csv
    csv.field_size_limit(10_000_000)
    path = download("AyonRoy29/BanglishDepYT", "unlabeled/youtube_collected_raw.csv")
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for r in take(csv.DictReader(fh)):
            offer(r.get("text"), "banglish_dep_yt", "bn")


def emonoba():
    import csv
    for f in ["Train.csv", "Val.csv"]:                 # Test.csv stays out (it is a public benchmark)
        path = download("khondoker/EmoNoBa", f)
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            for r in take(csv.DictReader(fh)):
                offer(r.get("Data") or r.get("text") or next(iter(r.values())), "emonoba", "bn")


def daraz_kaggle():
    import csv
    import glob
    for path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if "daraz" not in path.lower():
            continue
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            for r in take(csv.DictReader(fh)):
                text = next((v for k, v in r.items() if k and re.search("review|text|comment", k, re.I) and v), "")
                offer(text, "daraz", "bn")


# ---- Hindi / Hinglish
def hindi_tweets():
    repo = "sumanpaudel1997/hindi_tweet_sampled_dataset"
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    for t in take(parquet_texts(repo, files, "text")):
        offer(t, "hindi_tweets", "hi", names=True)
        if full("hi"):
            return


def wrestlers_kaggle():
    import csv
    import glob
    csv.field_size_limit(10_000_000)
    for path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if "wrestl" not in path.lower():
            continue
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            for r in take(csv.DictReader(fh)):
                offer(r.get("textOriginal") or r.get("textDisplay"), "wrestlers_yt", "hi")


def yt45k():
    import csv
    path = download("atakshat11/YouTube-Comments-Dataset-45k-rows",
                    "YouTube Comments Dataset with Sentiment, Toxicity, and Spam Labels (45K Rows).csv")
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for r in take(csv.DictReader(fh)):
            offer(r.get("comment_text"), "yt45k")


# ---- English (plenty: capped per emoji, quota)
def shorts():
    path = download("PingVortex/Youtube_shorts_comments", "shorts_comments.json")
    with open(path, "rb") as fh:
        for t in take(ijson.items(fh, "item.text")):
            offer(t, "yt_shorts")


def bluesky2m():
    repo = "alpindale/two-million-bluesky-posts"
    for f in sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".jsonl")):
        path = download(repo, f)
        for t in take(jsonl_texts(path)):
            offer(t, "bluesky_2m")
        os.remove(path)
        if full("en"):
            return


def twitter100m():
    repo = "enryu43/twitter100m_tweets"
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    for f in files[: (1 if SMOKE else 14)]:
        for t in take(parquet_texts(repo, [f], "tweet")):
            offer(t, "twitter100m")
        log("  twitter100m", f, {l: len(v) for l, v in kept.items()})
        if full("en"):
            return


def bluesky10m():
    path = download("itsmebatuhan/bluesky-10m-posts-15-languages", "bluesky_10M_post.jsonl")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for k, line in enumerate(fh):
            if LIMIT and k >= LIMIT:
                break
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("lang") in ("en", "hi", "bn"):
                offer(r.get("text"), "bluesky_10m", r["lang"] if r["lang"] != "en" else None)
            if full("en"):
                break
    os.remove(path)


for name, fn in [("banglishrev", banglishrev), ("bn_youtube", bn_youtube), ("coke_studio", coke_studio),
                 ("banglish_dep_yt", banglish_dep_yt), ("emonoba", emonoba), ("daraz", daraz_kaggle),
                 ("wrestlers_yt", wrestlers_kaggle), ("yt45k", yt45k), ("hindi_tweets", hindi_tweets),
                 ("yt_shorts", shorts), ("bluesky_2m", bluesky2m), ("twitter100m", twitter100m),
                 ("bluesky_10m", bluesky10m)]:
    run(name, fn)

# ---------------------------------------------------------------- split + write
test, train = [], []
for lang, rows in kept.items():
    random.shuffle(rows)
    n_test = min(2000 if not SMOKE else 200, len(rows) // 10)
    test += rows[:n_test]
    train += rows[n_test:]
random.shuffle(train)


def write(name, rows):
    with gzip.open(os.path.join(OUT, name), "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


write("train.jsonl.gz", train)
write("test.jsonl.gz", test)

cover = collections.Counter(i for r in train for i in r["e"])
summary = {
    "train": len(train), "test": len(test),
    "per_language": {l: len(v) for l, v in kept.items()},
    "per_source": {s: dict(c) for s, c in stats.items()},
    "emojis_with_1+": sum(1 for i in range(len(EMOJIS)) if cover[i] >= 1),
    "emojis_with_50+": sum(1 for i in range(len(EMOJIS)) if cover[i] >= 50),
    "top_emojis": {l: "".join(EMOJIS[i] for i, _ in first_count[l].most_common(30)) for l in kept},
    "samples": {l: [(r["t"][:120], "".join(EMOJIS[i] for i in r["e"])) for r in kept[l][:25]] for l in kept},
    "minutes": round((time.time() - T0) / 60, 1),
}
json.dump(summary, open(os.path.join(OUT, "stats.json"), "w"), ensure_ascii=False, indent=1)
log(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False, indent=1))
for l in kept:
    for t, e in summary["samples"][l][:8]:
        print(l, e, "<-", t)
