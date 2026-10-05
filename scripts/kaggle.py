#!/usr/bin/env python3
"""Tiny Kaggle API client (stdlib only). Reads KAGGLE_API_TOKEN / KAGGLE_USERNAME from .env.

  python3 scripts/kaggle.py push training/kaggle/laya_check.py emojet-laya-check [--gpu] [--internet] [--public]
  (a .ipynb file is pushed as a notebook, anything else as a script; --source=user/kernel and
  --dataset=user/dataset attach inputs under /kaggle/input)
  python3 scripts/kaggle.py status emojet-laya-check
  python3 scripts/kaggle.py wait   emojet-laya-check          # polls until done, then fetches output
  python3 scripts/kaggle.py output emojet-laya-check [dest] [--only=reference --only=.json]   # log + output files -> data/kaggle/<slug>/

Notebooks are pushed as private scripts unless --public is given. The token is never printed.
"""
import http.client
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://api.kaggle.com/v1/kernels.KernelsApiService/"


def env():
    vals = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip("'\"")
    return vals["KAGGLE_API_TOKEN"], vals["KAGGLE_USERNAME"]


TOKEN, USER = env()


def call(method, body):
    req = urllib.request.Request(
        API + method,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "User-Agent": "emojet"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"Kaggle API {method}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}")


def push(path, slug, gpu, internet, public, sources=(), datasets=()):
    res = call("SaveKernel", {
        "slug": f"{USER}/{slug}",
        "newTitle": slug,
        "text": (ROOT / path).read_text(encoding="utf-8"),
        "language": "python",
        "kernelType": "notebook" if str(path).endswith(".ipynb") else "script",
        "isPrivate": not public,
        "enableGpu": gpu,
        "enableInternet": internet,
        "machineShape": "NvidiaTeslaT4" if gpu else None,
        "datasetDataSources": list(datasets), "kernelDataSources": list(sources), "competitionDataSources": [],
        "modelDataSources": [], "categoryIds": [],
    })
    if res.get("error"):
        sys.exit(f"push failed: {res['error']}")
    print(f"pushed version {res.get('versionNumber')} -> {res.get('url')}")


def status(slug):
    res = call("GetKernelSessionStatus", {"userName": USER, "kernelSlug": slug})
    return res.get("status", "?"), res.get("failureMessage")


def download(url, path, tries=8):
    """Stream to disk; resume with HTTP Range after a dropped connection."""
    part = path.with_name(path.name + ".part")
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "emojet"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if have and r.status != 206:
                    have = 0                                   # server ignored Range: start over
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            part.rename(path)
            return
        except (OSError, http.client.HTTPException) as e:
            print(f"  connection dropped ({e.__class__.__name__}), resuming… ({attempt + 1}/{tries})", flush=True)
            time.sleep(3)
    raise SystemExit(f"download failed: {path.name}")


def output(slug, dest=None, only=None):
    """Fetch the log and output files; `only` = substrings, a file is fetched if its name contains one."""
    dest = pathlib.Path(dest or ROOT / "data" / "kaggle" / slug)
    dest.mkdir(parents=True, exist_ok=True)
    res = call("ListKernelSessionOutput", {"userName": USER, "kernelSlug": slug})
    (dest / "log.txt").write_text(res.get("log") or "", encoding="utf-8")
    for f in res.get("files") or []:
        name = f.get("fileName") or f.get("name")
        url = f.get("url")
        if only and not any(o in name for o in only):
            continue
        if name and url:
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            download(url, dest / name)
            print("downloaded", name)
    print("log ->", dest / "log.txt")
    return dest


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    flags = {a for a in args if a.startswith("--")}
    pos = [a for a in args if not a.startswith("--")]
    if cmd == "push":
        push(pos[0], pos[1], "--gpu" in flags, "--internet" in flags, "--public" in flags,
             [a.split("=", 1)[1] for a in args if a.startswith("--source=")],
             [a.split("=", 1)[1] for a in args if a.startswith("--dataset=")])
    elif cmd == "status":
        print(*status(pos[0]))
    elif cmd == "output":
        only = [a.split("=", 1)[1] for a in args if a.startswith("--only=")]
        output(pos[0], pos[1] if len(pos) > 1 else None, only or None)
    elif cmd == "wait":
        t0 = time.time()
        while True:
            st, msg = status(pos[0])
            print(f"[{int(time.time() - t0):5d}s] {st}" + (f" — {msg}" if msg else ""), flush=True)
            if st.lower() not in ("queued", "running", "new", "kernel_worker_status_queued", "kernel_worker_status_running", "?"):
                break
            time.sleep(20)
        output(pos[0])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
