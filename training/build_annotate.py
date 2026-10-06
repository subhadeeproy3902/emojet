#!/usr/bin/env python3
"""Build the Kaggle annotation script (training/annotate.py + taxonomy + emoji list).

  python3 training/build_annotate.py [--smoke]
  python3 scripts/kaggle.py push training/kaggle/emojet_annotate.py emojet-annotate --gpu --internet --source=subhadeep3902/emojet-data
  round 2: build with --round2, push as emojet-annotate-2 with --source=subhadeep3902/emojet-data
      --source=subhadeep3902/emojet-slang --source=subhadeep3902/emojet-annotate (round-1 labels: skipped; also has the prebuilt engine)
"""
import base64
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE = "--smoke" in sys.argv
ROUND2 = "--round2" in sys.argv
tax = json.loads((ROOT / "labels/tones.json").read_text(encoding="utf-8"))
tax["emojis"] = [e["e"] for e in json.loads((ROOT / "labels/emoji-catalog.json").read_text(encoding="utf-8"))["emojis"]]
b64 = base64.b64encode(gzip.compress(json.dumps(tax, ensure_ascii=False).encode())).decode()
commit = next(l.split('"')[1] for l in (ROOT / "training/build_notebook.py").read_text().splitlines() if l.startswith("LLAMA_CPP_COMMIT"))
src = (ROOT / "training/annotate.py").read_text(encoding="utf-8")
src = src.replace("__SMOKE__", str(SMOKE)).replace("__ROUND2__", str(ROUND2)).replace("__TAXONOMY__", b64).replace("__LLAMA_CPP_COMMIT__", commit)
out = ROOT / "training/kaggle" / ("emojet_annotate" + ("_round2" if ROUND2 else "") + ("_smoke" if SMOKE else "") + ".py")
out.write_text(src, encoding="utf-8")
print(f"wrote {out} (smoke={SMOKE}, llama.cpp {commit[:8]}, {len(tax['tones'])} tones)")
