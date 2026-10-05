#!/usr/bin/env python3
"""Build labels/emoji-catalog.json from the official Unicode emoji list + CLDR keywords.

Sources (downloaded into data/unicode/ by this script, Unicode License v3):
  https://unicode.org/Public/emoji/latest/emoji-test.txt
  cldr-json: cldr-annotations-full (en) + cldr-annotations-derived-full (en)

The catalog keeps one entry per *base* emoji. Skin-tone variants are not separate
entries: the app applies the user's skin-tone preference to emojis marked "tone".
"""
import json
import pathlib
import re
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "unicode"
SOURCES = {
    "emoji-test.txt": "https://unicode.org/Public/emoji/latest/emoji-test.txt",
    "cldr-annotations-en.json": "https://raw.githubusercontent.com/unicode-org/cldr-json/main/cldr-json/cldr-annotations-full/annotations/en/annotations.json",
    "cldr-annotationsDerived-en.json": "https://raw.githubusercontent.com/unicode-org/cldr-json/main/cldr-json/cldr-annotations-derived-full/annotationsDerived/en/annotations.json",
}
TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}
LINE = re.compile(r"^([0-9A-F ]+?)\s*;\s*fully-qualified\s*#\s*(\S+)\s+E(\d+\.\d+)\s+(.+)$")


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (DATA / name).exists():
            urllib.request.urlretrieve(url, DATA / name)


def cldr():
    out = {}
    for name in ("cldr-annotations-en.json", "cldr-annotationsDerived-en.json"):
        d = json.loads((DATA / name).read_text(encoding="utf-8"))
        ann = d.get("annotations") or d.get("annotationsDerived")
        out.update(ann["annotations"] if "annotations" in ann else ann)
    return out


def main():
    fetch()
    keywords = cldr()
    version = next(l for l in (DATA / "emoji-test.txt").read_text(encoding="utf-8").splitlines() if l.startswith("# Version"))

    entries, tone_bases, total, group, subgroup = [], set(), 0, "", ""
    for line in (DATA / "emoji-test.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
        elif line.startswith("# subgroup:"):
            subgroup = line.split(":", 1)[1].strip()
        elif m := LINE.match(line):
            total += 1
            char, name = m.group(2), m.group(4)
            if any(t in char for t in TONES):
                tone_bases.add("".join(c for c in char if c not in TONES))
                continue
            k = keywords.get(char) or keywords.get(char.replace("️", "")) or {}
            entries.append({
                "e": char,
                "name": name,
                "group": group,
                "sub": subgroup,
                "since": float(m.group(3)),
                "kw": [w for w in k.get("default", []) if w.lower() != name.lower()],
            })

    for e in entries:
        if e["e"] in tone_bases or e["e"].replace("️", "") in tone_bases:
            e["tone"] = True

    out = {
        "source": f"Unicode emoji-test.txt ({version.lstrip('# ').strip()}) + CLDR annotations, Unicode License v3",
        "fully_qualified_total": total,
        "base_count": len(entries),
        "emojis": entries,
    }
    path = ROOT / "labels" / "emoji-catalog.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    groups = {}
    for e in entries:
        groups[e["group"]] = groups.get(e["group"], 0) + 1
    print(f"{total} fully-qualified emojis -> {len(entries)} base entries "
          f"({sum(1 for e in entries if e.get('tone'))} support skin tones), "
          f"{sum(1 for e in entries if not e['kw'])} without keywords, {path.stat().st_size // 1024} KB")
    for g, n in groups.items():
        print(f"  {n:5d}  {g}")


if __name__ == "__main__":
    main()
