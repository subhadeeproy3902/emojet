#!/usr/bin/env python3
"""Record llama.cpp's Laya answers for a fixed set of requests, to check other runtimes against.

Run against a llama-server started with the BF16 model:
  MODEL=models/Laya-BF16.gguf bash scripts/dev-server.sh &
  python3 training/make_reference.py      # -> training/reference/llamacpp-laya-bf16.json
"""
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8080/v1/systemone"

moods = json.loads((ROOT / "labels/moods.json").read_text(encoding="utf-8"))
tones = json.loads((ROOT / "labels/tones.json").read_text(encoding="utf-8"))

QUESTIONS = {
    "mood": {"type": "choice", "instructions": moods["question"], "criteria": {k: None for k in moods["moods"]}},
    "tone": {"type": "choice", "instructions": tones["question"], "criteria": {k: None for k in tones["tones"]}},
    "style": {"type": "choice", "instructions": tones["styles"]["question"],
              "criteria": {k: v["desc"] for k, v in tones["styles"]["options"].items()}},
    "sarcastic": {"type": "noul", "instructions": "Is my message sarcastic?"},
    "energy": {"type": "score", "instructions": "How energetic is my message?", "criteria": ["flat", "calm", "lively", "hyped"]},
}
STATES = [
    "Happy birthday bro!! have an amazing year",
    "oh great, another monday meeting that could have been an email",
    "Fine, I'll do it your way, since you always know best.",
    {"replying_to": "do you want to come to the concert with me?", "my_draft": "yes!!"},
]


def main():
    cases = []
    for state in STATES:
        body = {"state": state, "questions": QUESTIONS}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        cases.append({"request": body, "answers": out["answers"], "usage": out["usage"]})
        print(state if isinstance(state, str) else state["my_draft"], "->",
              {q: a.get("choice", a.get("noul", a.get("score"))) for q, a in out["answers"].items()})
    dest = ROOT / "training/reference/llamacpp-laya-bf16.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"runtime": "llama.cpp llama-server, Laya-BF16.gguf", "cases": cases},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
