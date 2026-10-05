#!/usr/bin/env python3
"""Build the Kaggle script that collects real emoji-labelled messages (training/real_data.py + the emoji list).

  python3 training/build_data.py [--smoke]
  python3 scripts/kaggle.py push training/kaggle/emojet_data.py emojet-data --internet \
      --dataset=aatmanvaidya/youtube-comments-on-wrestlers-protest --dataset=rrokon/daraz-bd-reviews-bengalibanglish-nlp
"""
import base64
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE = "--smoke" in sys.argv
emojis = [e["e"] for e in json.loads((ROOT / "labels/emoji-catalog.json").read_text(encoding="utf-8"))["emojis"]]
b64 = base64.b64encode(gzip.compress(json.dumps(emojis, ensure_ascii=False).encode())).decode()
src = (ROOT / "training/real_data.py").read_text(encoding="utf-8")
src = src.replace("__SMOKE__", str(SMOKE)).replace("__CATALOG__", b64)
out = ROOT / "training/kaggle" / ("emojet_data_smoke.py" if SMOKE else "emojet_data.py")
out.write_text(src, encoding="utf-8")
print(f"wrote {out} ({len(emojis)} emojis, smoke={SMOKE})")
