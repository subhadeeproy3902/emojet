#!/usr/bin/env python3
"""Build the "model soup" notebook: blend two trained Emojet models' weights and keep the best blend.

Weight averaging (WiSE-FT, Wortsman et al. 2022) keeps most of a fine-tune's new skill and most of the base
model's original accuracy. Round 3b learned slang/everyday Banglish-Hinglish (+17 probes) but lost 2-3 points on
real held-out messages; the blend old*(1-a) + new*a is evaluated for several a with the exact same evaluation as
training, and the best blend that is not worse than the shipped model is saved (weights + GGUF).

  python3 training/build_soup.py <commit> [--smoke]
  python3 scripts/kaggle.py push training/kaggle/emojet_soup.ipynb emojet-v2-soup --gpu --internet \\
      --source=subhadeep3902/emojet-v2-round2 --source=subhadeep3902/emojet-v2-round3b --source=subhadeep3902/emojet-data

It reuses the cells of the round-3 training notebook (settings, labels, evaluation, save), so build that first:
python3 training/build_notebook.py <commit> --round3
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMIT = sys.argv[1] if len(sys.argv) > 1 else "main"
SMOKE = "--smoke" in sys.argv
src_nb = json.loads((ROOT / "training/kaggle/emojet_v2_round3.ipynb").read_text(encoding="utf-8"))
code_cells = [c["source"] for c in src_nb["cells"] if c["cell_type"] == "code"]


def find(marker):
    return next(c for c in code_cells if marker in c)


settings, labels, data, evaluation, save = (find("LAYA_REV ="), find("def emoji_ids"), find("def ge_targets"),
                                            find("def evaluate("), find("save_pretrained"))
ge_fn = data[data.index("def ge_targets"):]
ge_fn = ge_fn[:ge_fn.index("\nfor r in ge[")]

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip()})


md(f"""
# Emojet — model soup (round 2 × round 3b)

Blends the weights of the shipped model (round 2) and round 3b (slang + everyday Banglish/Hinglish), evaluates each
blend with the training notebook's evaluation, and saves the best blend that is not worse than round 2.
Repo commit `{COMMIT[:10]}`; plan in `docs/TRAINING.md`.
""")
code(settings)
code(labels)
code(f"""
# evaluation data only (no training data)
from datasets import load_dataset
from collections import Counter
import glob, gzip as _gzip, html
import pandas as pd
from huggingface_hub import hf_hub_download
emz = load_dataset("yagizgencer/emojinize-multilingual", "all")
ge = load_dataset("google-research-datasets/go_emotions", "simplified")
GE_LABELS, GE = ge_map["labels"], ge_map["map"]
GE_TONES = sorted({{TONE_IDX[t] for v in GE.values() for t in v["tones"]}})
{ge_fn}
SARC = TONE_IDX["sarcastic"]
sarc_test = pd.read_excel(hf_hub_download("BM2002/hinglish-sarcasm-dataset", "testing_data_hinglish.xlsx", repo_type="dataset"))
rdir = os.path.dirname(sorted(glob.glob("/kaggle/input/**/real/train.jsonl.gz", recursive=True))[0])
real_test = [json.loads(l) for l in _gzip.open(f"{{rdir}}/test.jsonl.gz", "rt", encoding="utf-8")]
if {SMOKE}:
    real_test = real_test[::10]
print(len(real_test), "real test messages")
""")
code("""
from transformers import AutoTokenizer, ModernBertForSequenceClassification
dirs = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/emojet.json", recursive=True))
print(dirs)
DIR_OLD = next(d for d in dirs if "round2" in d)
DIR_NEW = next(d for d in dirs if "round3b" in d)
tok = AutoTokenizer.from_pretrained(DIR_NEW)
old = ModernBertForSequenceClassification.from_pretrained(DIR_OLD, torch_dtype=torch.float32)
model = ModernBertForSequenceClassification.from_pretrained(DIR_NEW, torch_dtype=torch.float32).cuda()
assert old.config.id2label == model.config.id2label, "the two models must have the same outputs in the same order"
assert model.config.num_labels == N_LABELS
SD_OLD = {k: v.detach().float().clone() for k, v in old.state_dict().items()}
SD_NEW = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
assert SD_OLD.keys() == SD_NEW.keys()
del old
NAME = "emojet-v2"
def set_alpha(a):
    model.load_state_dict({k: (1 - a) * SD_OLD[k] + a * SD_NEW[k] for k in SD_NEW})
""")
code(evaluation)
code(f"""
ALPHAS = {[0.0, 0.5, 1.0] if SMOKE else [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]}
results = {{}}
for a in ALPHAS:
    set_alpha(a)
    results[a] = evaluate()
    print(f"alpha {{a}}:", results[a], flush=True)

def num(v):
    if isinstance(v, str) and "/" in v:
        x, y = v.split("/"); return float(x) / max(1.0, float(y))
    return float(v)
ref = results[0.0]                                   # round 2 = the shipped model
REAL = [f"real_{{l}}_{{m}}" for l in ("en", "hi", "bn") for m in ("top1", "hit@6")]
def passes(r):
    # not worse than round 2: real messages within 0.5 points (2,000 per language: that is noise), chat cases,
    # emoji cases, tones within one case, style within one case, sarcasm within 1 point, lookalikes
    ok = all(num(r[k]) >= num(ref[k]) - 0.005 for k in REAL)
    ok &= num(r["chat_cases_top1"]) >= num(ref["chat_cases_top1"]) and num(r["chat_cases_top3"]) >= num(ref["chat_cases_top3"])
    ok &= num(r["emoji_cases_hit@6"]) >= num(ref["emoji_cases_hit@6"])
    ok &= num(r["tone_top3"]) >= num(ref["tone_top3"]) - 1 / 25 and num(r["style"]) >= num(ref["style"]) - 1 / 13
    ok &= num(r["hinglish_sarcasm_acc"]) >= num(ref["hinglish_sarcasm_acc"]) - 0.01
    ok &= num(r["slang_negatives_clean"]) >= num(ref["slang_negatives_clean"])
    return ok
def skill(r):
    return num(r["slang_hit@6"]) + num(r["slang_capitalised_hit@6"]) + num(r["slang_hit@3"]) + num(r["chat_cases_top1"])
keys = list(ref)
print(f"{{'metric':26s}}" + "".join(f"{{a:>9}}" for a in ALPHAS))
for k in keys:
    print(f"{{k:26s}}" + "".join(f"{{str(results[a][k]):>9s}}" for a in ALPHAS))
print(f"{{'passes (no regression)':26s}}" + "".join(f"{{str(passes(results[a])):>9s}}" for a in ALPHAS))
good = [a for a in ALPHAS if a > 0 and passes(results[a])]
if good:
    BEST = max(good, key=lambda a: skill(results[a]))
    print(f"best blend: alpha {{BEST}} (no regression vs round 2, most slang skill)")
else:
    BEST = None
    print("!! no blend is free of regressions vs round 2: nothing new to ship; saving the 0.5 blend for inspection")
CHOSEN = BEST if BEST is not None else 0.5
set_alpha(CHOSEN)
final, baseline, crash, history = results[CHOSEN], ref, None, [dict(alpha=a, **results[a]) for a in ALPHAS]
ROUND2, ROUND3 = False, True
""")
save_cell = save.replace('metrics=final,', 'metrics=final, soup=dict(alpha=CHOSEN, passes=BEST is not None, old=DIR_OLD, new=DIR_NEW),')
code(save_cell)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "training/kaggle" / ("emojet_soup" + ("_smoke" if SMOKE else "") + ".ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
