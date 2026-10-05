#!/usr/bin/env python3
"""Build the Kaggle script that collects human-labelled tone/style datasets (training/tone_data.py + taxonomy).

  python3 training/build_tone_data.py [--smoke]
  python3 scripts/kaggle.py push training/kaggle/emojet_tones.py emojet-tones --internet \
      --dataset=abushahadatrabby/ben-sarc-dataset --dataset=mdabdulkaium/bengali-humordataset
"""
import base64
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE = "--smoke" in sys.argv
tax = json.loads((ROOT / "labels/tones.json").read_text(encoding="utf-8"))
tax["emojis"] = [e["e"] for e in json.loads((ROOT / "labels/emoji-catalog.json").read_text(encoding="utf-8"))["emojis"]]
b64 = base64.b64encode(gzip.compress(json.dumps(tax, ensure_ascii=False).encode())).decode()
src = (ROOT / "training/tone_data.py").read_text(encoding="utf-8").replace("__SMOKE__", str(SMOKE)).replace("__TAXONOMY__", b64)
out = ROOT / "training/kaggle" / ("emojet_tones_smoke.py" if SMOKE else "emojet_tones.py")
out.write_text(src, encoding="utf-8")
print(f"wrote {out} (smoke={SMOKE})")
