#!/usr/bin/env python3
"""Simulate the app's emoji ranking on top of a running Emojet server and compare strategies.

  MODEL=models/Emojet-v1-Q4_K_M.gguf bash scripts/dev-server.sh &
  python3 scripts/rank_sim.py

"plain" = top-6 raw scores. "pooled" = what the app does (keep in sync with Ranker.kt):
reply blending, tone boost, topic guarantee with a bonus, vibe emojis excluded from the bonus.
"""
import json
import math
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8080/embedding"

cat = json.loads((ROOT / "labels/emoji-catalog.json").read_text(encoding="utf-8"))["emojis"]
import os
# label layout must match the model: Emojet-v1 / v2-pre have 46 tones (labels/tones-v1.json), v2 has 65 (labels/tones.json)
tones_cfg = json.loads((ROOT / os.environ.get("TONES", "labels/tones.json")).read_text(encoding="utf-8"))
EMOJIS = [e["e"] for e in cat]
TONES = list(tones_cfg["tones"])
NE, NT = len(EMOJIS), len(TONES)
TOPIC_GROUPS = {"Animals & Nature", "Food & Drink", "Travel & Places", "Activities", "Objects", "Flags"}
VIBE = set("🎉🎊✨🔥💥⭐🌟💫⚡🌈🎈🎆🎇🪩") | {"⭐"}
TOPIC_SUBS = {"face-costume", "person-role", "person-activity", "person-sport", "person-fantasy",
              "person-resting", "family", "zodiac", "religion", "warning", "transport-sign"}
# Emojis nobody wants suggested on an ordinary message ("Brishti porche baire" -> 🖕🤬): allowed only when the
# model itself reads the message as rude/angry/aggressive/derisive.
OFFENSIVE = {"🖕", "🤬", "🤮", "🤢", "🍆", "🍑", "👿"}
ANGRY_TONES = ("rude", "aggressive", "angry", "derisive")
OFFENSIVE_MIN = 0.5
IS_TOPIC = [(e["group"] in TOPIC_GROUPS or e["sub"] in TOPIC_SUBS) and e["e"] not in VIBE for e in cat]
E_INDEX = {e: i for i, e in enumerate(EMOJIS)}
TONE_ROWS = {TONES.index(t): [E_INDEX[x] for x in v["emojis"] if x in E_INDEX] for t, v in tones_cfg["tones"].items()}
SKIP_TONE_BOOST = {"neutral", "friendly"}

# knobs (mirrored in Ranker.kt)
TOPIC_WINDOW = 4.5      # a topic emoji competes if within this many nats of the best emoji
TOPIC_BONUS = 1.6       # nats added to competing topic emojis
MAX_TOPIC = 3
TONE_MIN = 0.5
TONE_BONUS = 0.8


def fmt(draft, reply=None):
    return (f"replying to: {reply.strip()}\n" if reply else "") + f"my message: {draft.strip()}"


def scores(text):
    req = urllib.request.Request(URL, data=json.dumps({"content": text, "embd_normalize": -1}).encode(),
                                 headers={"Content-Type": "application/json"})
    v = json.loads(urllib.request.urlopen(req).read())[0]["embedding"]
    while isinstance(v[0], list):
        v = v[0]
    return v


OFF_IDX = [i for i, e in enumerate(EMOJIS) if e in OFFENSIVE]
ANGRY_IDX = [i for i, t in enumerate(TONES) if t in ANGRY_TONES]


def angry(s):
    return max((1 / (1 + math.exp(-s[NE + t])) for t in ANGRY_IDX), default=0.0)


def guard(adj, s):
    """Drop 🖕🤬… unless the model reads the message itself as rude/angry."""
    if angry(s) < OFFENSIVE_MIN:
        for i in OFF_IDX:
            adj[i] = -1e9
    return adj


def plain(draft, reply=None):
    s = scores(fmt(draft, reply))
    return sorted(range(NE), key=lambda i: -guard(list(s[:NE]), s)[i])[:6], s


def pooled(draft, reply=None):
    s = scores(fmt(draft))
    if reply:
        # a confident draft ("Laptop") keeps its own emojis; a vague one ("yes!!") follows the reply
        m1 = max(s[:NE]); z1 = sum(math.exp(x - m1) for x in s[:NE]); conf = 1 / z1
        w = min(0.85, max(0.25, 0.25 + conf * 1.5))
        s2 = scores(fmt(draft, reply))
        s = [w * a + (1 - w) * b for a, b in zip(s, s2)]
    m = max(s[:NE])
    z = math.log(sum(math.exp(x - m) for x in s[:NE]))
    lp = [x - m - z for x in s[:NE]]
    adj = lp[:]
    for t in range(NT):
        p = 1 / (1 + math.exp(-s[NE + t]))
        if p >= TONE_MIN and TONES[t] not in SKIP_TONE_BOOST:
            for i in TONE_ROWS[t]:
                adj[i] += TONE_BONUS * p
    best = max(lp)
    topic = sorted((i for i in range(NE) if IS_TOPIC[i] and lp[i] >= best - TOPIC_WINDOW), key=lambda i: -lp[i])[:MAX_TOPIC]
    for i in topic:
        adj[i] += TOPIC_BONUS
    return sorted(range(NE), key=lambda i: -guard(adj, s)[i])[:6], s


def norm(e):
    return "".join(c for c in e if c != "️" and not 0x1F3FB <= ord(c) <= 0x1F3FF)


def main():
    sets = json.loads((ROOT / "training/data/eval_sets.json").read_text(encoding="utf-8"))
    for name, fn in (("plain", plain), ("pooled", pooled)):
        hits, t0 = 0, time.time()
        for c in sets["emoji_cases"]:
            top, _ = fn(c["text"], c["reply"])
            ok = {norm(x) for x in c["ok_emojis"]}
            hits += any(norm(EMOJIS[i]) in ok for i in top)
        print(f"{name:7s} emoji cases hit@6: {hits}/{len(sets['emoji_cases'])}  ({(time.time() - t0) / len(sets['emoji_cases']) * 1000:.0f} ms each)")
        h1 = h3 = 0
        for c in sets.get("chat_cases", []):
            top, _ = fn(c["text"], c["reply"])
            ok = {norm(x) for x in c["ok_emojis"]}
            h1 += norm(EMOJIS[top[0]]) in ok
            h3 += any(norm(EMOJIS[i]) in ok for i in top[:3])
        print(f"{name:7s} chat cases top-1: {h1}/{len(sets.get('chat_cases', []))}  top-3: {h3}/{len(sets.get('chat_cases', []))}")
        # slang / everyday Banglish-Hinglish probes (round 3) and lookalikes that must NOT get toilet/swear emojis
        sp, sn = sets.get("slang_probes", []), sets.get("slang_negatives", [])
        if sp:
            h6, h3, per, tot, miss = 0, 0, {}, {}, []
            for c in sp:
                top, _ = fn(c["text"], c.get("reply"))
                ok = {norm(x) for x in c["ok_emojis"]}
                a6 = any(norm(EMOJIS[i]) in ok for i in top)
                h6 += a6; h3 += any(norm(EMOJIS[i]) in ok for i in top[:3])
                per[c["lang"]] = per.get(c["lang"], 0) + a6; tot[c["lang"]] = tot.get(c["lang"], 0) + 1
                if not a6: miss.append(f"{c['text']} → {''.join(EMOJIS[i] for i in top)}")
            langs = "  ".join(f"{l} {per[l]}/{tot[l]}" for l in sorted(tot))
            print(f"{name:7s} slang probes hit@6: {h6}/{len(sp)}  hit@3: {h3}/{len(sp)}  ({langs})")
            for m in miss: print(f"          ✗ {m}")
        if sn:
            bad = []
            for c in sn:
                top, _ = fn(c["text"])
                if {norm(EMOJIS[i]) for i in top} & {norm(x) for x in c["bad_emojis"]}:
                    bad.append(f"{c['text']} → {''.join(EMOJIS[i] for i in top)}")
            print(f"{name:7s} lookalikes clean: {len(sn) - len(bad)}/{len(sn)}" + "".join(f"\n          ✗ {b}" for b in bad))
    print()
    probes = [("Program", None), ("Laptop", None), ("code", None), ("I'm writing code all night", None),
              ("my program finally works", None), ("debugging this bug for hours", None), ("need a new laptop", None),
              ("just adopted a puppy!!", None), ("I'm watching television", None), ("let's get pizza tonight?", None),
              ("Laptop", "are you coming to the party tonight?"), ("yes!!", "do you want to come to the concert with me?"),
              ("oh great, another monday meeting that could have been an email", None),
              ("Bhai ami khete jacchi, pore kotha bolchi", None), ("i miss you so much", None),
              ("Robot is perfect", None), ("Ekdom", None), ("Haga pacche", None), ("Tor matha thik hoe geche", None),
              ("Ami bhalo e achi bhai", None)]
    for d, r in probes:
        a, _ = plain(d, r)
        b, _ = pooled(d, r)
        print(f"plain {''.join(EMOJIS[i] for i in a):14s} | pooled {''.join(EMOJIS[i] for i in b):14s} ← {d!r}" + (f" ↪ {r!r}" if r else ""))


if __name__ == "__main__":
    main()
