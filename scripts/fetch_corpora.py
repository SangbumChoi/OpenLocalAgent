#!/usr/bin/env python
"""Download the pretraining corpora by explicit path through the mirror.

The cluster reaches Hugging Face only through a mirror that serves `resolve` URLs but not the
listing API, so the file list travels with the job instead of being discovered on the box.
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="pretrain_corpora.json")
ap.add_argument("--out", default="data/pretrain-src")
args = ap.parse_args()

endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
plan = json.loads(Path(args.plan).read_text())
for tag, entry in plan.items():
    target = Path(args.out) / tag
    target.mkdir(parents=True, exist_ok=True)
    got = bytes_total = 0
    started = time.time()
    for name in entry["files"]:
        destination = target / name.replace("/", "__")
        if destination.exists() and destination.stat().st_size > 0:
            got += 1
            bytes_total += destination.stat().st_size
            continue
        url = f"{endpoint}/datasets/{entry['repo']}/resolve/main/{name}"
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = response.read()
        except Exception as error:
            print(f"  FAIL {tag} {name}: {type(error).__name__} {str(error)[:90]}", flush=True)
            continue
        destination.write_bytes(payload)
        got += 1
        bytes_total += len(payload)
    print(f"{tag:16s} files={got}/{len(entry['files'])} "
          f"bytes={bytes_total/1e6:8.1f}MB secs={time.time()-started:.0f}", flush=True)
print("FETCH_CORPORA_DONE", flush=True)
