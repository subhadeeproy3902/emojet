#!/usr/bin/env python3
"""Check a Laya-Emoji GGUF running in llama-server against the Kaggle notebook's reference outputs,
then score it on the shared test sets.

  MODEL=models/Laya-Emoji-Q4_0.gguf bash scripts/dev-server.sh &
  python3 scripts/test-laya-emoji.py [--reference data/kaggle/<run>/reference.json]
"""
import argparse
import json
import math
import pathlib
import statistics
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()), (time.perf_counter() - t0) * 1000


def fmt(draft, reply=None):
    return (f"replying to: {reply.strip()}\n" if reply else "") + f"my message: {draft.strip()}"


def norm(e):
    return "".join(c for c in e if c != "️" and not 0x1F3FB <= ord(c) <= 0x1F3FF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()

    emojis = [e["e"] for e in json.loads((ROOT / "labels/emoji-catalog.json").read_text(encoding="utf-8"))["emojis"]]
    tones_cfg = json.loads((ROOT / "labels/tones.json").read_text(encoding="utf-8"))
    tones, styles = list(tones_cfg["tones"]), list(tones_cfg["styles"]["options"])
    NE, NT, NS = len(emojis), len(tones), len(styles)

    def scores(text):
        out, ms = post(f"{args.url}/embedding", {"content": text, "embd_normalize": -1})
        v = out[0]["embedding"]
        while isinstance(v[0], list):
            v = v[0]
        return v, ms

    def summary(v):
        top = sorted(range(NE), key=lambda i: -v[i])[:6]
        tp = sorted(range(NT), key=lambda i: -v[NE + i])[:3]
        st = max(range(NS), key=lambda i: v[NE + NT + i])
        return [emojis[i] for i in top], [(tones[i], round(1 / (1 + math.exp(-v[NE + i])), 2)) for i in tp], styles[st]

    if args.reference:
        print("== llama.cpp vs PyTorch (reference.json)")
        for ref in json.loads(pathlib.Path(args.reference).read_text(encoding="utf-8")):
            toks, _ = post(f"{args.url}/tokenize", {"content": ref["text"], "add_special": True})
            ids = toks["tokens"] if isinstance(toks, dict) else toks
            v, _ = scores(ref["text"])
            diff = max(abs(a - b) for a, b in zip(v, ref["logits"]))
            e, t, s = summary(v)
            print(f"  tokens {'same' if ids == ref['input_ids'] else 'DIFFERENT'} ({len(ids)} vs {len(ref['input_ids'])}), "
                  f"max score diff {diff:.3f}, top emojis {''.join(e)} vs {''.join(ref['top_emojis'])}  | {ref['text'][:50]!r}")

    sets = json.loads((ROOT / "training/data/eval_sets.json").read_text(encoding="utf-8"))
    print("\n== emoji cases (hit = an acceptable emoji among the 6)")
    hits, times = 0, []
    for c in sets["emoji_cases"]:
        v, ms = scores(fmt(c["text"], c["reply"]))
        times.append(ms)
        e, t, s = summary(v)
        ok = {norm(x) for x in c["ok_emojis"]}
        h = any(norm(x) in ok for x in e)
        hits += h
        print(f"  {'✓' if h else '✗'} {ms:5.0f}ms {c['name']:18s} {''.join(e)}  {[x for x, _ in t]}")
    print(f"emoji hit@6: {hits}/{len(sets['emoji_cases'])}, median {statistics.median(times):.0f} ms")

    print("\n== tone cases")
    th = sh = sn = 0
    for c in sets["tone_cases"]:
        v, _ = scores(fmt(c["text"]))
        e, t, s = summary(v)
        tn = {x for x, _ in t}
        th += bool(tn & set(c["ok_tones"]))
        if c["ok_styles"]:
            sn += 1
            sh += s in c["ok_styles"]
        print(f"  {'✓' if tn & set(c['ok_tones']) else '✗'} {s:18s} {[x for x, _ in t]}  {''.join(e)}  {c['text'][:45]}")
    print(f"tone top-3: {th}/{len(sets['tone_cases'])}   style: {sh}/{sn}")


if __name__ == "__main__":
    main()
