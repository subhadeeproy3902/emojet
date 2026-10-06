#!/usr/bin/env python3
"""Build the Kaggle script that collects slang/swear lexicons and real slang messages (training/slang_data.py).

  python3 training/build_slang.py [--smoke]
  python3 scripts/kaggle.py push training/kaggle/emojet_slang.py emojet-slang --internet --source=subhadeep3902/emojet-data
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
src = (ROOT / "training/slang_data.py").read_text(encoding="utf-8").replace("__SMOKE__", str(SMOKE)).replace("__TAXONOMY__", b64)
out = ROOT / "training/kaggle" / ("emojet_slang_smoke.py" if SMOKE else "emojet_slang.py")
out.write_text(src, encoding="utf-8")
print(f"wrote {out} (smoke={SMOKE})")
