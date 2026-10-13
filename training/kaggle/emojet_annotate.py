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

SMOKE = False
TAXONOMY_B64 = "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"
LLAMA_CPP_COMMIT = "2e7c58c5477478c8cf6e199cfaa5dcd5a4319c81"
MODEL_REPO, MODEL_FILE = "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TIME_BUDGET_H = 0.6 if SMOKE else 6.0
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
for lang, rows in pool.items():
    random.shuffle(rows)
    msgs += rows[:PER_LANG[lang]]
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
emojis (shown in brackets) they are strong evidence.

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


stat = {"batches": 0, "tokens": 0, "secs": 0.0, "cut": 0}


def ask(batch):
    lines = []
    for k, x in enumerate(batch, 1):
        typed = "".join(EMOJIS[i] for i in x.get("e") or [])
        lines.append(f"{k}. {x['t'][:500]}" + (f"   [{typed}]" if typed else ""))
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
