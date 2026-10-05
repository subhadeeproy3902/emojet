#!/usr/bin/env python3
"""Score Laya's zero-shot communication-style and tone-of-voice answers (labels/tones.json)."""
import argparse
import json
import pathlib
import statistics
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (message, acceptable styles or None if ambiguous, acceptable tones)
SAMPLES = [
    ("I understand your point, but I need us to meet this deadline first.", {"assertive"}, {"authoritative", "formal", "confident", "respectful", "serious"}),
    ("Whatever everyone else wants to do is fine with me.", {"passive"}, {"apathetic", "tentative", "neutral", "friendly"}),
    ("You need to stop doing it that way because it's wrong.", {"aggressive"}, {"aggressive", "accusatory", "condescending", "demanding", "disapproving", "authoritative"}),
    ("Fine, I'll do it your way, since you always know best.", {"passive-aggressive"}, {"sarcastic", "bitter", "cynical", "defensive", "condescending"}),
    ("oh wow, another meeting, my favourite thing ever", {"passive-aggressive"}, {"sarcastic", "cynical"}),
    ("That's so sweet of you, thank you for remembering my birthday", None, {"appreciative", "loving", "joyful"}),
    ("I'm so sorry, I completely forgot to call you back", None, {"apologetic"}),
    ("You can do this, I believe in you!", {"assertive"}, {"encouraging", "confident", "empathetic"}),
    ("That must have been really hard for you, I'm here if you need anything", None, {"empathetic"}),
    ("Honestly I'm not sure that's how it works, but maybe?", {"passive"}, {"tentative", "skeptical", "confused"}),
    ("Well, obviously you wouldn't understand, it's a bit complex for you", {"aggressive", "passive-aggressive"}, {"condescending"}),
    ("It wasn't my fault, I did exactly what you told me to do", None, {"defensive", "angry", "accusatory"}),
    ("You always mess this up and then blame everyone else", {"aggressive"}, {"accusatory", "angry", "aggressive"}),
    ("meh, do whatever, I don't really care", {"passive"}, {"apathetic"}),
    ("WE GOT THE TICKETS!!! I'm literally screaming", None, {"enthusiastic", "joyful"}),
    ("Please submit the report by 5 PM today. No exceptions.", {"assertive", "aggressive"}, {"authoritative", "demanding", "formal", "urgent"}),
    ("Dear team, please find the updated schedule attached.", {"assertive"}, {"formal", "neutral", "respectful"}),
    ("The train leaves at 6:40 from platform 3.", None, {"neutral"}),
    ("hehe you wish you could beat me at this game", None, {"playful", "confident", "witty"}),
    ("remember when we used to bunk classes and eat maggi at 2am", None, {"nostalgic", "joyful"}),
    ("i'm kinda nervous about the interview tomorrow", None, {"anxious"}),
    ("wait WHAT, she said yes??", None, {"surprised", "enthusiastic"}),
    ("people only call you when they need something, that's life", None, {"cynical", "bitter", "pessimistic"}),
    ("you looked really cute today ngl", None, {"flirty", "admiring", "loving"}),
    ("it's fine. really. I'm fine.", {"passive-aggressive", "passive"}, {"bitter", "sad", "sarcastic", "defensive", "apathetic"}),
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/systemone")
    ap.add_argument("--style-desc", action="store_true", help="send the style descriptions")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "labels" / "tones.json").read_text(encoding="utf-8"))
    styles = {k: (v["desc"] if args.style_desc else None) for k, v in cfg["styles"]["options"].items()}
    tones = {k: None for k in cfg["tones"]}
    questions = {
        "style": {"type": "choice", "instructions": cfg["styles"]["question"], "criteria": styles},
        "tone": {"type": "choice", "instructions": cfg["question"], "criteria": tones},
    }
    print(f"{len(styles)} styles (descriptions {'on' if args.style_desc else 'off'}), {len(tones)} tones\n")

    s_ok = s_n = t1 = t3 = 0
    times = []
    for text, ok_styles, ok_tones in SAMPLES:
        out, ms = post(args.url, {"state": text, "questions": questions})
        times.append(ms)
        a = out["answers"]
        tp = a["tone"]["probabilities"]
        ranked = sorted(tp, key=lambda k: -tp[k])[:3]
        h1, h3 = ranked[0] in ok_tones, bool(ok_tones & set(ranked))
        t1, t3 = t1 + h1, t3 + h3
        style = a["style"]["choice"]
        smark = " "
        if ok_styles:
            s_n += 1
            s_ok += style in ok_styles
            smark = "✓" if style in ok_styles else "✗"
        tmark = "✓" if h1 else ("~" if h3 else "✗")
        tones_s = " ".join(f"{t}:{tp[t]:.2f}" for t in ranked)
        print(f"{tmark}{smark} {ms:5.0f}ms {out['usage']['input_tokens']:4d}tok  {style:18s} {tones_s:48s} {text[:48]}")

    n = len(SAMPLES)
    print(f"\ntone top-1: {t1}/{n}  top-3: {t3}/{n}   style: {s_ok}/{s_n}")
    print(f"latency (2 questions): median {statistics.median(times):.0f} ms")


if __name__ == "__main__":
    main()
