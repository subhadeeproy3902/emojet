# EmoJet — Phase T step 1, runs on Kaggle (free T4 GPU). Pushed with: python3 scripts/kaggle.py push ...
# 1. Original Laya in PyTorch (its own reference code) vs llama.cpp's BF16 answers (training/reference/).
# 2. Laya's encoder loaded into ModernBertForSequenceClassification = the Laya-Emoji model shape.
# 3. Training speed + GPU memory for that model, to size the real training run.
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"  # convaiinnovations/laya, same as the GGUF source
REF_URL = "https://raw.githubusercontent.com/subhadeeproy3902/emojet/main/training/reference/llamacpp-laya-bf16.json"
N_LABELS = 1923 + 46 + 4 + 4  # emojis + tones + styles + tone dimensions
OUT = {}


def log(key, value):
    OUT[key] = value
    print(f"{key} = {value}", flush=True)


def section(name, fn):
    t0 = time.time()
    try:
        fn()
        log(f"{name}_ok", True)
    except Exception:
        log(f"{name}_ok", False)
        log(f"{name}_error", traceback.format_exc()[-1500:])
    log(f"{name}_seconds", round(time.time() - t0, 1))


def tv(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "0"


if tuple(int(x) for x in tv("transformers").split(".")[:2]) < (4, 48):  # ModernBERT needs >= 4.48
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.48"], check=True)

import torch  # noqa: E402
import transformers  # noqa: E402

log("torch", torch.__version__)
log("transformers", transformers.__version__)
log("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
log("gpu_mem_GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0)

S = {}


def download():
    from huggingface_hub import snapshot_download
    S["dir"] = snapshot_download("convaiinnovations/laya", revision=REV,
                                 allow_patterns=["*.json", "*.py", "model.safetensors", "tokenizer/*", "encoder/*"])
    log("files", sorted(os.listdir(S["dir"])))


def compare():
    sys.path.insert(0, S["dir"])
    from rl_agent_api import RLAgent
    agent = RLAgent(S["dir"])
    S["agent"] = agent
    ref = json.loads(urllib.request.urlopen(REF_URL, timeout=60).read())
    worst, same_n, total, rows = 0.0, 0, 0, []
    for case in ref["cases"]:
        req = case["request"]
        got = agent.system_one(req["state"], req["questions"])
        for qid, exp in case["answers"].items():
            g = got["answers"][qid]
            if exp["type"] == "noul":
                diff = abs(g["noul"] - exp["noul"])
                same = (g["noul"] > 0.5) == (exp["noul"] > 0.5)
                gv, ev = round(g["noul"], 3), round(exp["noul"], 3)
            else:
                diff = max(abs(g["probabilities"][k] - v) for k, v in exp["probabilities"].items())
                if exp["type"] == "choice":
                    same, gv, ev = g["choice"] == exp["choice"], g["choice"], exp["choice"]
                else:
                    same, gv, ev = abs(g["score"] - exp["score"]) < 0.25, round(g["score"], 2), round(exp["score"], 2)
            worst, same_n, total = max(worst, diff), same_n + same, total + 1
            rows.append([str(req["state"])[:28], qid, round(diff, 4), same, gv, ev])
        log(f"tokens_pytorch_vs_llamacpp_{len(rows)}", [got["usage"]["input_tokens"], case["usage"]["input_tokens"]])
    log("reference_same_answer", f"{same_n}/{total}")
    log("reference_worst_probability_diff", round(worst, 4))
    log("reference_rows [state, question, diff, same, pytorch, llama.cpp]", rows)


def classifier():
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, ModernBertForSequenceClassification
    d = S["dir"]
    sd = load_file(os.path.join(d, "model.safetensors"))
    log("checkpoint_tensors", len(sd))
    log("checkpoint_prefixes", sorted({k.split(".")[0] for k in sd}))
    cfg = AutoConfig.from_pretrained(os.path.join(d, "encoder"))
    cfg.num_labels = N_LABELS
    cfg.problem_type = "multi_label_classification"
    clf = ModernBertForSequenceClassification(cfg)
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    res = clf.model.load_state_dict(enc, strict=False)
    log("encoder_tensors_loaded", len(enc))
    log("encoder_missing_keys", res.missing_keys)
    log("encoder_unexpected_keys", res.unexpected_keys)
    log("classifier_params_M", round(sum(p.numel() for p in clf.parameters()) / 1e6, 1))
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    x = tok(["it's fine. really. I'm fine."], return_tensors="pt")
    x = {k: v.cuda() for k, v in x.items() if k in ("input_ids", "attention_mask")}
    clf = clf.cuda().eval()
    with torch.no_grad():
        a = S["agent"].model.encoder.float()(**x).last_hidden_state.float()
        b = clf.model(**x).last_hidden_state.float()
    log("encoder_hidden_state_max_diff", float((a - b).abs().max()))
    S["clf"] = clf


def train_speed():
    S.pop("agent", None)
    torch.cuda.empty_cache()
    clf = S["clf"].train()
    opt = torch.optim.AdamW(clf.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda")
    for B, L in ((32, 64), (64, 48)):
        ids = torch.randint(5, 50000, (B, L), device="cuda")
        am = torch.ones_like(ids)
        y = torch.zeros(B, N_LABELS, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(8):
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = clf(input_ids=ids, attention_mask=am).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        sps = B / (sum(times[3:]) / len(times[3:]))
        log(f"train_b{B}_len{L}_samples_per_sec", round(sps, 1))
        log(f"train_b{B}_len{L}_peak_mem_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
        log(f"train_b{B}_len{L}_minutes_per_epoch_of_200k", round(200_000 / sps / 60, 1))


for name, fn in (("download", download), ("compare", compare), ("classifier", classifier), ("train_speed", train_speed)):
    section(name, fn)
    if not OUT.get(f"{name}_ok") and name in ("download",):
        break

os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/result.json", "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print("DONE")
