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
import subprocess
import sys
import threading
import time
import urllib.request

SMOKE = __SMOKE__
TAXONOMY_B64 = "__TAXONOMY__"
LLAMA_CPP_COMMIT = "__LLAMA_CPP_COMMIT__"
MODEL_REPO, MODEL_FILE = "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TIME_BUDGET_H = 0.6 if SMOKE else 6.0
PER_LANG = {"bn": 400, "hi": 400, "en": 400} if SMOKE else {"bn": 70_000, "hi": 45_000, "en": 45_000}
BATCH, SLOTS = 16, 8

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
if not os.path.exists("/tmp/llama.cpp/build/bin/llama-server"):
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
                         f"-DCMAKE_BUILD_TYPE=Release > /tmp/cmake.log 2>&1 && "
                         f"cmake --build build --target llama-server -j 4 > /tmp/build.log 2>&1", shell=True)
    if res.returncode:
        for f in ("/tmp/cmake.log", "/tmp/build.log"):
            if os.path.exists(f):
                txt = open(f, errors="replace").read()
                print(f"==== {f}\n" + "\n".join(l for l in txt.splitlines() if "rror" in l or "fatal" in l)[-3000:] + "\n" + txt[-1500:])
        raise SystemExit("llama.cpp build failed")
log("llama-server built")
from huggingface_hub import hf_hub_download  # noqa: E402
model_path = hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir="/tmp/model")
log("model downloaded", round(os.path.getsize(model_path) / 2**30, 1), "GB")
server = subprocess.Popen(
    ["/tmp/llama.cpp/build/bin/llama-server", "-m", model_path, "--host", "127.0.0.1", "--port", "8080",
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

STYLE (exactly one, the writer's behavior toward the reader):
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

Answer with JSON only."""

SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {
        "i": {"type": "integer"},
        "tones": {"type": "array", "items": {"enum": TONES}, "minItems": 1, "maxItems": 3},
        "style": {"enum": STYLES},
        "axes": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 9}, "minItems": 6, "maxItems": 6},
        "emojis": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}},
    "required": ["i", "tones", "style", "axes", "emojis"]}}}, "required": ["items"]}


def ask(batch):
    lines = []
    for k, x in enumerate(batch, 1):
        typed = "".join(EMOJIS[i] for i in x.get("e") or [])
        lines.append(f"{k}. {x['t'][:500]}" + (f"   [{typed}]" if typed else ""))
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Annotate these messages:\n" + "\n".join(lines)}],
            "temperature": 0.0, "max_tokens": 90 * len(batch),
            "response_format": {"type": "json_schema", "json_schema": {"name": "annotations", "schema": SCHEMA}}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    items = json.loads(out["choices"][0]["message"]["content"])["items"]
    res = []
    for it in items:
        k = it.get("i", 0) - 1
        if 0 <= k < len(batch):
            res.append({**batch[k], "tones": it["tones"], "style": it["style"], "axes": it["axes"], "emojis": it["emojis"]})
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
        if k % 50 == 0:
            rate = done / max(1, time.time() - t_run)
            log(f"{done:,} annotated, {failed:,} failed, {rate * 3600:,.0f}/hour")
fh.close()
server.terminate()
log(f"finished: {done:,} annotated, {failed:,} failed")
rows = [json.loads(l) for l in open(os.path.join(OUT, "annotations.jsonl"), encoding="utf-8")]
for r in rows[:30]:
    print(r["l"], "|", r["t"][:90], "|", r["tones"], r["style"], r["axes"], "".join(r["emojis"]))
from collections import Counter  # noqa: E402
print("tones:", Counter(t for r in rows for t in r["tones"]).most_common())
print("styles:", Counter(r["style"] for r in rows).most_common())
