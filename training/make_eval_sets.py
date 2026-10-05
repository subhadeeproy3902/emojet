#!/usr/bin/env python3
"""Write training/data/eval_sets.json from the same samples the laptop tests use (never trained on)."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


moods = json.loads((ROOT / "labels/moods.json").read_text(encoding="utf-8"))["moods"]
emoji_cases = [{"name": name, "text": text, "reply": reply,
                "ok_emojis": sorted({e for m in ok for e in moods[m]["emojis"]})}
               for name, text, reply, ok in load("test-prompts").SAMPLES]
emoji_cases.append({"name": "banglish-eat-later", "text": "Bhai ami khete jacchi, pore kotha bolchi", "reply": None,
                    "ok_emojis": sorted(set("🍛🍚😋🍽️🍔🍕🍜🍲🤤👋🏃📞⏰✌️🙏") - {"️"})})
tone_cases = [{"text": text, "ok_styles": sorted(ok_s) if ok_s else None, "ok_tones": sorted(ok_t)}
              for text, ok_s, ok_t in load("test-tones").SAMPLES]
out = ROOT / "training/data/eval_sets.json"
out.write_text(json.dumps({"emoji_cases": emoji_cases, "tone_cases": tone_cases}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(emoji_cases)} emoji cases, {len(tone_cases)} tone cases -> {out}")
