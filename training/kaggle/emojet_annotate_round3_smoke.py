"""Emojet tone/style annotation (runs on Kaggle, 2x T4, internet on).

Real messages (from the emojet-data kernel) are labelled with the full taxonomy in labels/tones.json
(65 tones, 8 behavioral styles, 6 sliders) plus the emojis a writer would likely add, by an open
model (Qwen3.6-35B-A3B, Apache-2.0) on llama.cpp. Output is constrained by a JSON schema, so only
valid label names come back. Results are appended as they arrive (a timeout loses nothing).

Output: /kaggle/working/annot/annotations.jsonl   {"t", "l", "s", "e" (writer's emojis), "tones", "style", "axes", "emojis"}

Built by training/build_annotate.py, which fills in TAXONOMY_B64, SMOKE and the llama.cpp commit.
"""
import base64
import concurrent.futures as cf
import gzip
import glob
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request

SMOKE = True
ROUND2 = False        # round 2: real messages that use slang/swearing (emojet-slang) + more Bengali/Hindi, skip round-1 texts
ROUND3 = True        # round 3: cleaned slang/chat-word messages (emojet-slang-3) WITH word meanings + synthetic chat lines
CHATLEX_B64 = "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"
CHAT_LEXICON_B64 = "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"
TAXONOMY_B64 = "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"
LLAMA_CPP_COMMIT = "2e7c58c5477478c8cf6e199cfaa5dcd5a4319c81"
MODEL_REPO, MODEL_FILE = "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TIME_BUDGET_H = 0.6 if SMOKE else (4.0 if ROUND3 else (3.0 if ROUND2 else 6.0))
SYNTH_BUDGET_H = 0.15 if SMOKE else 1.25   # round 3: the synthetic phase runs first and stops here
PER_LANG = {"bn": 400, "hi": 400, "en": 400} if SMOKE else {"bn": 45_000, "hi": 25_000, "en": 20_000}
BATCH, SLOTS = 12, 16

T0 = time.time()
OUT = "/kaggle/working/annot"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


TAX = json.loads(gzip.decompress(base64.b64decode(TAXONOMY_B64)).decode())
TONES = list(TAX["tones"])
STYLES = list(TAX["styles"]["options"])
AXES = [k for k in TAX["dimensions"] if k != "note"]
EMOJIS = TAX["emojis"]

# ---------------------------------------------------------------- messages
random.seed(7)
rdir = os.path.dirname(sorted(glob.glob("/kaggle/input/**/real/train.jsonl.gz", recursive=True))[0])
pool = {"en": [], "hi": [], "bn": []}
for name in ("train.jsonl.gz", "plain.jsonl.gz"):
    path = os.path.join(rdir, name)
    if os.path.exists(path):
        for line in gzip.open(path, "rt", encoding="utf-8"):
            r = json.loads(line)
            if r["l"] in pool and 3 <= len(r["t"]) <= 500:
                pool[r["l"]].append(r)
msgs = []
_ns = {}
if ROUND3:
    exec(gzip.decompress(base64.b64decode(CHATLEX_B64)).decode("utf-8"), _ns)
    LEXICON = json.loads(gzip.decompress(base64.b64decode(CHAT_LEXICON_B64)).decode("utf-8"))["entries"]
    LX = _ns["Lexicon"](LEXICON)
if ROUND3:
    before = set()
    for path in glob.glob("/kaggle/input/**/annot/annotations.jsonl", recursive=True):
        for line in open(path, encoding="utf-8"):
            try:
                before.add(json.loads(line)["t"])
            except ValueError:
                pass
    targets = []
    for path in glob.glob("/kaggle/input/**/slang/targets.jsonl.gz", recursive=True):
        rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
        if rows and "notes" in rows[0]:                       # round-3 format only (round 2's targets were half junk)
            targets += rows
    targets = [r for r in targets if r["t"] not in before]
    # round-robin over words, most useful first: curated Bengali/Hindi chat words, curated English, then swear lists.
    # So if time runs out, every word has had its share instead of the first few words taking everything.
    def prio(r):
        return (0 if r.get("curated") else 2) + (0 if r["l"] in ("bn", "hi") else 1)
    by_term = {}
    for r in sorted(targets, key=prio):
        by_term.setdefault((prio(r), r["term"]), []).append(r)
    order = []
    for tier in sorted({k[0] for k in by_term}):
        lists = [v for k, v in by_term.items() if k[0] == tier]
        for i in range(max(map(len, lists), default=0)):
            order += [v[i] for v in lists if i < len(v)]
    msgs = order[: (300 if SMOKE else 45_000)]
    log(f"round 3: {len(targets):,} new target messages over {len(by_term):,} words; annotating up to {len(msgs):,}")
elif ROUND2:
    before = set()
    for path in glob.glob("/kaggle/input/**/annot/annotations.jsonl", recursive=True):
        for line in open(path, encoding="utf-8"):
            try:
                before.add(json.loads(line)["t"])
            except ValueError:
                pass
    slang = []
    for path in glob.glob("/kaggle/input/**/slang/targets.jsonl.gz", recursive=True):
        slang += [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    slang = [r for r in slang if r["t"] not in before]
    # A "slang" word that is common in ordinary messages is just a normal word (lists include "na", "bhai", "kya"):
    # drop terms found in more than 0.2% of a 100k sample of real Bengali/Hindi messages. The owner's terms are kept.
    KEEP = {"haga", "hagu", "haga pacche", "ekdom", "ekdum", "matha kharap", "pagol", "bokachoda", "jhakaas", "bc", "mc",
            "chutiya", "bsdk", "bkl"}
    terms = {r.get("term", "") for r in slang if r["s"] != "slang-english"}
    single = {t for t in terms if t and " " not in t and t.isascii()}
    multi = [t for t in terms if " " in t and t.isascii()]
    script = [t for t in terms if t and not t.isascii()]
    script_re = re.compile("|".join(sorted(map(re.escape, script), key=len, reverse=True))) if script else None
    sample = random.sample(pool["bn"] + pool["hi"], min(100_000, len(pool["bn"]) + len(pool["hi"])))
    df = {}
    for r in sample:
        low = r["t"].lower()
        words = re.findall(r"[a-z]+", low)
        found = set(words) & single
        if multi:
            joined = " " + " ".join(words) + " "
            found |= {t for t in multi if " " + t + " " in joined}
        if script_re:
            found |= set(script_re.findall(low))
        for t in found:
            df[t] = df.get(t, 0) + 1
    common = {t for t, n in df.items() if n / len(sample) > 0.002 and t not in KEEP}
    log(f"slang terms: {len(terms):,}, dropped as common words: {len(common)}:", sorted(common, key=lambda t: -df[t])[:80])
    slang = [r for r in slang if r.get("term") not in common]
    slang.sort(key=lambda r: {"bn": 0, "hi": 1, "en": 2}.get(r["l"], 3))      # Banglish/Bengali first, English slang last
    msgs = slang[: (300 if SMOKE else 32_000)]
    taken = before | {r["t"] for r in msgs}
    for lang, n in (("bn", 100 if SMOKE else 10_000), ("hi", 100 if SMOKE else 4_000)):
        rows = [r for r in pool[lang] if r["t"] not in taken]
        random.shuffle(rows)
        msgs += rows[:n]
    log(f"round 2: {len(before):,} done before, {len(slang):,} new slang messages")
else:
    for lang, rows in pool.items():
        random.shuffle(rows)
        msgs += rows[:PER_LANG[lang]]
if not ROUND3:                    # round 3 keeps its word-by-word round-robin order
    random.shuffle(msgs)
log("to annotate:", len(msgs), {l: min(len(v), PER_LANG[l]) for l, v in pool.items()})

# ---------------------------------------------------------------- engine
# a llama-server built by an earlier run of this kernel (attached as input) saves ~25 GPU minutes
SERVER_BIN = next(iter(sorted(glob.glob("/kaggle/input/**/bin/llama-server-cuda", recursive=True))), None)
if SERVER_BIN:
    subprocess.run(["cp", SERVER_BIN, "/tmp/llama-server-cuda"], check=True)
    os.chmod("/tmp/llama-server-cuda", 0o755)
    SERVER_BIN = "/tmp/llama-server-cuda"
    log("using prebuilt llama-server")
    os.makedirs("/kaggle/working/bin", exist_ok=True)              # pass it on to the next run
    subprocess.run(["cp", SERVER_BIN, "/kaggle/working/bin/llama-server-cuda"], check=True)
if not SERVER_BIN:
    subprocess.run(f"git clone -q https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && cd /tmp/llama.cpp && "
                   f"git checkout -q {LLAMA_CPP_COMMIT}", shell=True, check=True)
    nvcc = next((p for p in ("/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc") if os.path.exists(p)), "nvcc")
    subprocess.run("nvidia-smi; " + nvcc + " --version; cmake --version", shell=True)
    # Kaggle's CUDA image has no libcuda.so where CMake looks: point it at the stub or the driver library
    libcuda = next((p for pat in ("/usr/local/cuda/lib64/stubs/libcuda.so", "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so",
                                  "/usr/lib/x86_64-linux-gnu/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                                  "/usr/local/nvidia/lib64/libcuda.so.1", "/usr/lib64/libcuda.so.1")
                    for p in glob.glob(pat)), None)
    print("libcuda:", libcuda, flush=True)
    res = subprocess.run(f"cd /tmp/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
                         f"-DCMAKE_CUDA_COMPILER={nvcc} {'-DCUDA_cuda_driver_LIBRARY=' + libcuda if libcuda else ''} -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
                         f"-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF > /tmp/cmake.log 2>&1 && "
                         f"cmake --build build --target llama-server -j 4 > /tmp/build.log 2>&1", shell=True)
    if res.returncode:
        for f in ("/tmp/cmake.log", "/tmp/build.log"):
            if os.path.exists(f):
                txt = open(f, errors="replace").read()
                print(f"==== {f}\n" + "\n".join(l for l in txt.splitlines() if "rror" in l or "fatal" in l)[-3000:] + "\n" + txt[-1500:])
        raise SystemExit("llama.cpp build failed")
    SERVER_BIN = "/tmp/llama.cpp/build/bin/llama-server"
    os.makedirs("/kaggle/working/bin", exist_ok=True)              # keep it for the next run (static binary)
    subprocess.run(["cp", SERVER_BIN, "/kaggle/working/bin/llama-server-cuda"], check=True)
log("llama-server built")
from huggingface_hub import hf_hub_download  # noqa: E402
model_path = hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir="/tmp/model")
log("model downloaded", round(os.path.getsize(model_path) / 2**30, 1), "GB")
server = subprocess.Popen(
    [SERVER_BIN, "-m", model_path, "--host", "127.0.0.1", "--port", "8080",
     "-ngl", "999", "-ts", "1,1", "-np", str(SLOTS), "-c", str(SLOTS * 4096), "-fa", "on", "--jinja",
     "--chat-template-kwargs", '{"enable_thinking": false}', "--no-webui", "-b", "2048", "-ub", "512"],
    stdout=open("/tmp/server.log", "w"), stderr=subprocess.STDOUT)
for _ in range(600):
    try:
        if b"ok" in urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5).read():
            break
    except Exception:
        pass
    if server.poll() is not None:
        print(open("/tmp/server.log").read()[-3000:])
        raise SystemExit("llama-server exited")
    time.sleep(2)
log("server up")

# ---------------------------------------------------------------- prompt
def tone_line(name, v):
    return f"- {name}: {', '.join(v.get('aliases', [])[:5]) or name}"


SYSTEM = f"""You are an expert annotator of chat messages for tone of voice, communication style and emoji use.
Messages are real chat/social text in English, Hindi (Devanagari or romanized Hinglish) or Bengali (script or romanized
Banglish), often with slang, typos and code-mixing. Read each message as the people who wrote it would mean it: understand
slang, idioms, sarcasm and banter in all three languages. Judge only the writer's tone in this message.

TONES (pick 1 to 3 that clearly apply, most dominant first; use "neutral" only if nothing else fits):
{chr(10).join(tone_line(n, v) for n, v in TAX['tones'].items())}

STYLE (the writer's behavior toward the reader; "-" if the message shows no clear style, e.g. plain small talk or a statement):
{chr(10).join(f'- {n}: {v["desc"]}' for n, v in TAX['styles']['options'].items())}

AXES (integers 0..9, in this order):
1 funny(0) .. serious(9)
2 formal(0) .. casual(9)
3 respectful(0) .. irreverent(9)
4 enthusiastic(0) .. matter-of-fact(9)
5 high-context: implicit, needs shared history to understand(0) .. low-context: explicit and literal(9)
6 linear: structured, one point(0) .. circular: narrative, digressive(9)

EMOJIS: up to 4 emojis this writer would most likely put in the message, most likely first. Pick what a real person in
this chat would send (the thing mentioned, the feeling, the joke), not generic decoration. If the writer already used
emojis (shown in brackets) they are strong evidence. Some messages end with (meaning: ...) explaining slang or regional
words: use it. Include the emoji for the thing the message is about (needing the toilet -> 🚽💩, rain -> 🌧️, sleepy -> 😴)
together with the feeling people actually add (crude jokes usually get 😂). Swear words between friends are often banter.

FORMAT: exactly one line per message, nothing else:
<number>|<tones, comma-separated>|<style or ->|<the 6 axes as 6 digits>|<emojis>
Example lines (for other messages):
1|sarcastic,playful|passive-aggressive|274635|🙄😂
2|joyful,appreciative|-|382723|🥳❤️🙏"""

ALIAS = {}
for _n, _v in TAX["tones"].items():
    ALIAS[_n] = _n
    for _a in _v.get("aliases", []):
        ALIAS.setdefault(_a.lower(), _n)
LINE = re.compile(r"^\s*(\d+)\s*\|([^|]*)\|([^|]*)\|\s*(\d{6})\s*\|(.*)$")


def parse(line, n):
    m = LINE.match(line)
    if not m:
        return None
    k = int(m.group(1)) - 1
    tones = []
    for t in m.group(2).split(","):
        t = ALIAS.get(t.strip().lower())
        if t and t not in tones:
            tones.append(t)
    style = m.group(3).strip().lower()
    style = style if style in STYLES else None
    emojis = [ch for ch in m.group(5).strip().split() if ch] or [m.group(5).strip()]
    if not 0 <= k < n or not tones:
        return None
    return k, tones[:3], style, [int(c) for c in m.group(4)], "".join(emojis)[:40]




def ask(batch):
    lines = []
    for k, x in enumerate(batch, 1):
        typed = "".join(EMOJIS[i] for i in x.get("e") or [])
        notes = "; ".join(x.get("notes") or [])
        lines.append(f"{k}. {x['t'][:500]}" + (f"   [{typed}]" if typed else "") + (f"   (meaning: {notes})" if notes else ""))
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Annotate these messages:\n" + "\n".join(lines)}],
            "temperature": 0.0, "max_tokens": 60 * len(batch) + 50}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    ch = out["choices"][0]
    stat["batches"] += 1; stat["tokens"] += out.get("usage", {}).get("completion_tokens", 0); stat["secs"] += time.time() - t
    if ch.get("finish_reason") != "stop":
        stat["cut"] += 1
    res, got = [], set()
    for line in ch["message"]["content"].splitlines():
        p = parse(line, len(batch))
        if p and p[0] not in got:
            got.add(p[0])
            k, tones, style, axes, emojis = p
            res.append({**batch[k], "tones": tones, "style": style, "axes": axes, "emojis": [emojis]})
    return res


# ---------------------------------------------------------------- round 3: synthetic chat lines for the lexicon
# Real Banglish/Hinglish chat about everyday things (sleepy, rain, toilet) is rare in public data (reviews, comments,
# tweets). The model writes short WhatsApp-style lines that use each word; a line is kept only if the matcher finds
# that word in it in the right language. Weighted as silver data in training; never used for evaluation.
SCRIPTS = {"bn": [("banglish", "Banglish: Bengali typed in English letters, the way people type on WhatsApp", "roman"),
                  ("bengali", "Bengali script", "native")],
           "hi": [("hinglish", "Hinglish: Hindi typed in English letters, the way people type on WhatsApp", "roman"),
                  ("hindi", "Hindi in Devanagari script", "native")],
           "en": [("english", "casual English as young people text", "roman")]}
SITUATIONS = ["friends' group chat", "texting a sibling or cousin", "texting a partner", "family group chat", "office or college friends",
              "late night chat", "making plans", "complaining", "joking around", "replying to a friend's message"]
SYNTH_SYSTEM = ("You write realistic chat messages exactly the way real people in India and Bangladesh type on WhatsApp: "
                "casual, short, natural spelling, code-mixing where natural. No translations, no explanations, no numbering.")


def synth(entry, variety, desc, form, rnd):
    terms = entry[form] or []
    if not terms:
        return []
    term = rnd.choice(terms)
    sit = ", ".join(rnd.sample(SITUATIONS, 3))
    user = (f"Write 12 different chat messages in {desc} that naturally use \"{term}\" (meaning: {entry['gloss']}). "
            f"Mix very short ones (1-4 words) and longer ones (up to 25 words). Situations: {sit}. Vary the mood. "
            f"One message per line, then ' || ' and 1 to 4 emojis the writer would add (the thing the message is about and "
            f"the feeling). Format:\n<message> || <emojis>")
    body = {"messages": [{"role": "system", "content": SYNTH_SYSTEM}, {"role": "user", "content": user}],
            "temperature": 0.9, "top_p": 0.95, "seed": rnd.randint(0, 10**9), "max_tokens": 900}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"]
    rows = []
    for line in out.splitlines():
        if "||" not in line:
            continue
        text, emo = (x.strip() for x in line.rsplit("||", 1))
        text = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", text).strip().strip('"')
        if not 2 <= len(text) <= 200 or not emo:
            continue
        lang = entry["lang"]
        if not any(LEXICON[k] is entry for k, _ in LX.find(text, lang)):
            stat["synth_rejected"] += 1
            continue
        rows.append({"t": text, "l": lang, "s": "synth-" + variety, "e": [], "emojis": [emo[:40]], "term": term,
                     "gloss": entry["gloss"], "kind": entry["kind"]})
    return rows


stat = {"batches": 0, "tokens": 0, "secs": 0.0, "cut": 0, "synth_rejected": 0}
if ROUND3:
    jobs = []
    for i, e in enumerate(LEXICON):
        if e["kind"] == "address":
            continue
        for variety, desc, form in SCRIPTS[e["lang"]]:
            for rep_ in range(1 if SMOKE else 2):
                jobs.append((e, variety, desc, form, random.Random(i * 100 + rep_ * 7 + len(variety))))
    if SMOKE:
        jobs = jobs[:12]
    random.shuffle(jobs)
    t_s, n_syn = time.time(), 0
    sfh = open(os.path.join(OUT, "synthetic.jsonl"), "a", encoding="utf-8")
    slock = threading.Lock()

    def sjob(j):
        global n_syn
        if (time.time() - T0) / 3600 > SYNTH_BUDGET_H:
            return
        try:
            rows = synth(*j)
        except Exception as exc:
            log("synth failed:", repr(exc)[:150])
            return
        with slock:
            for r in rows:
                sfh.write(json.dumps(r, ensure_ascii=False) + "\n")
            sfh.flush()
            n_syn += len(rows)

    with cf.ThreadPoolExecutor(SLOTS) as pool_:
        for k, f in enumerate(cf.as_completed([pool_.submit(sjob, j) for j in jobs]), 1):
            if k % (4 if SMOKE else 50) == 0:
                log(f"synthetic: {k}/{len(jobs)} requests, {n_syn:,} lines kept, {stat['synth_rejected']:,} rejected (word missing)")
    sfh.close()
    log(f"synthetic done: {n_syn:,} lines in {(time.time() - t_s) / 60:.0f} min")

# ---------------------------------------------------------------- run
lock = threading.Lock()
done = failed = 0
fh = open(os.path.join(OUT, "annotations.jsonl"), "a", encoding="utf-8")


def work(batch):
    global done, failed
    if (time.time() - T0) / 3600 > TIME_BUDGET_H:
        return
    for attempt in range(2):
        try:
            res = ask(batch)
            with lock:
                for r in res:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                done += len(res)
                failed += len(batch) - len(res)
            return
        except Exception as exc:
            if attempt:
                with lock:
                    failed += len(batch)
                log("batch failed:", repr(exc)[:200])


batches = [msgs[i:i + BATCH] for i in range(0, len(msgs), BATCH)]
t_run = time.time()
with cf.ThreadPoolExecutor(SLOTS) as ex:
    futs = [ex.submit(work, b) for b in batches]
    for k, f in enumerate(cf.as_completed(futs), 1):
        if k % (10 if SMOKE else 100) == 0:
            rate = done / max(1, time.time() - t_run)
            log(f"{done:,} annotated, {failed:,} failed, {rate * 3600:,.0f}/hour; "
                f"{stat['tokens'] / max(1, stat['batches']):.0f} tokens/batch, {stat['cut']} cut off")
fh.close()
server.terminate()
log(f"finished: {done:,} annotated, {failed:,} failed")
rows = [json.loads(l) for l in open(os.path.join(OUT, "annotations.jsonl"), encoding="utf-8")]
for r in rows[:30]:
    print(r["l"], "|", r["t"][:90], "|", r["tones"], r["style"], r["axes"], "".join(r["emojis"]))
from collections import Counter  # noqa: E402
print("tones:", Counter(t for r in rows for t in r["tones"]).most_common())
print("styles:", Counter(r["style"] for r in rows).most_common())
