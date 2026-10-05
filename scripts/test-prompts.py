#!/usr/bin/env python3
"""Send sample chat messages to a running llama-server (/v1/systemone) and score Laya's mood picks."""
import argparse
import json
import pathlib
import statistics
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (name, draft, message being replied to, moods a human would accept)
SAMPLES = [
    ("happy-birthday", "Happy birthday bro!! have an amazing year", None, {"celebration"}),
    ("lol", "hahaha that was the funniest thing I've seen all week", None, {"laughing"}),
    ("sad", "I didn't get the job, feeling really low today", None, {"sad"}),
    ("thanks", "thank you so much for helping me move yesterday", None, {"grateful"}),
    ("angry", "this is the third time the delivery guy didn't show up, I'm so done", None, {"angry"}),
    ("love", "I miss you so much, can't wait to see you", None, {"love"}),
    ("sarcasm", "oh great, another monday meeting that could have been an email", None, {"sarcastic", "angry", "tired"}),
    ("food", "let's get pizza tonight?", None, {"food"}),
    ("tired", "worked 14 hours today, I'm dead", None, {"tired"}),
    ("congrats", "congratulations on the promotion, you deserve it!", None, {"proud", "celebration"}),
    ("late", "running 10 mins late sorry", None, {"late", "sorry"}),
    ("goodnight", "good night, sleep well", None, {"goodnight"}),
    ("flirty", "you looked really cute today", None, {"flirty", "love"}),
    ("scared", "there's a huge spider in my bathroom help", None, {"scared"}),
    ("plain", "ok see you at 5", None, {"agree", "neutral", "greeting"}),
    ("hype", "we won the match!!! let's gooo", None, {"hype", "celebration", "proud"}),
    ("sick", "I've got a fever and a sore throat", None, {"sick"}),
    ("travel", "landed in Goa finally, beach time", None, {"travel"}),
    ("hinglish", "bhai kya scene hai aaj party karte hain", None, {"celebration", "hype"}),
    ("cringe", "I accidentally sent that text to my boss", None, {"awkward", "scared"}),
    ("reply-same", "same here", "I just finished the whole series in one night", {"agree", "tired", "laughing"}),
    ("reply-where", "where exactly?", "let's meet near the station", {"neutral", "travel", "agree"}),
    ("reply-yes", "yes!!", "do you want to come to the concert with me?", {"hype", "agree", "celebration", "love"}),
]


def load_moods():
    return json.loads((ROOT / "labels" / "moods.json").read_text(encoding="utf-8"))


def build_request(cfg, text, reply_to, with_desc, sarcasm):
    state = {"replying_to": reply_to, "my_draft": text} if reply_to else text
    criteria = {k: (v["desc"] if with_desc else None) for k, v in cfg["moods"].items()}
    questions = {"mood": {"type": "choice", "instructions": cfg["question"], "criteria": criteria}}
    if sarcasm:
        questions["sarcasm"] = {"type": "noul", "instructions": "Is my message sarcastic?"}
    return {"state": state, "questions": questions}


def suggest(cfg, mood_probs, n=6, top_moods=3):
    """Blend the emoji rows of the most likely moods. Same algorithm the apps will use."""
    scores = {}
    for mood, p in sorted(mood_probs.items(), key=lambda kv: -kv[1])[:top_moods]:
        for rank, emoji in enumerate(cfg["moods"][mood]["emojis"]):
            scores[emoji] = scores.get(emoji, 0.0) + p / (1 + rank * 0.35)
    return [e for e, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:n]]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/systemone")
    ap.add_argument("--desc", action="store_true", help="also send the short mood descriptions (slower, no better in Phase 0)")
    ap.add_argument("--no-reply", action="store_true", help="drop the replied-to message (to measure reply-awareness)")
    ap.add_argument("--sarcasm", action="store_true", help="also ask the sarcasm yes/no question (one more pass)")
    args = ap.parse_args()

    cfg = load_moods()
    print(f"{len(cfg['moods'])} moods, descriptions={'on' if args.desc else 'off'}, "
          f"reply context={'off' if args.no_reply else 'on'}\n")
    post(args.url, build_request(cfg, "warm up", None, args.desc, args.sarcasm))

    times, hit1, hit3 = [], 0, 0
    for name, text, reply, ok in SAMPLES:
        reply = None if args.no_reply else reply
        out, ms = post(args.url, build_request(cfg, text, reply, args.desc, args.sarcasm))
        probs = out["answers"]["mood"]["probabilities"]
        ranked = sorted(probs, key=lambda k: -probs[k])
        h1, h3 = ranked[0] in ok, bool(ok & set(ranked[:3]))
        hit1, hit3 = hit1 + h1, hit3 + h3
        times.append(ms)
        mark = "✓" if h1 else ("~" if h3 else "✗")
        moods = " ".join(f"{m}:{probs[m]:.2f}" for m in ranked[:3])
        line = f"{mark} {ms:5.0f}ms {out['usage']['input_tokens']:3d}tok {name:14s} {''.join(suggest(cfg, probs))}  {moods}"
        if args.sarcasm:
            line += f"  sarc={out['answers']['sarcasm']['noul']:.2f}"
        print(line)

    n = len(SAMPLES)
    print(f"\ntop-1 mood correct: {hit1}/{n}   in top-3: {hit3}/{n}")
    print(f"latency ms: median {statistics.median(times):.0f}, min {min(times):.0f}, max {max(times):.0f}")


if __name__ == "__main__":
    main()
