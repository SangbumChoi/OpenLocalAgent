#!/usr/bin/env python
"""How much of each suite's gold tool vocabulary a training corpus already contains.

A suite whose gold names all appear in training measures recall of a memorised mapping; one whose
names are unseen measures reading the catalog. The two are different claims, so the overlap is
worth knowing before quoting either.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openlocalagent.eval.suite import SUITES, build_tasks

def corpus_names(paths):
    called, catalog = set(), set()
    rows = 0
    for rel in paths:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print("MISSING", rel)
            continue
        for line in open(p):
            if not line.strip():
                continue
            rows += 1
            d = json.loads(line)
            for t in (d.get("tools") or []):
                n = t.get("name") if isinstance(t, dict) else None
                if n:
                    catalog.add(n)
            for m in (d.get("messages") or []):
                for tc in (m.get("tool_calls") or []):
                    n = tc.get("name")
                    if n:
                        called.add(n)
    return called, catalog, rows

UNION = ["data/merged-v2/train.jsonl"]
FIVE = ["data/distill2/train-clean.jsonl", "data/public/mobileactions-train.jsonl",
        "data/public/toucan-train.jsonl", "data/public/toolbench-train.jsonl",
        "data/public/device-train.jsonl"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default=".", help="corpus root; paths below are relative to it")
args = parser.parse_args()
root = args.root

u_called, u_cat, u_rows = corpus_names(UNION)
f_called, f_cat, f_rows = corpus_names(FIVE)
print(f"union      rows={u_rows:6d} distinct called={len(u_called):5d} catalog={len(u_cat):5d}")
print(f"five-source rows={f_rows:6d} distinct called={len(f_called):5d} catalog={len(f_cat):5d}")
print()
print(f"{'suite':14s} {'goldN':>6s} {'union called':>13s} {'union cat':>10s} {'5src called':>12s} {'5src cat':>9s}")
out = {}
for name in ("mobileactions", "xlam", "toolbench", "toolace", "bfcl", "androidcontrol", "agentnet", "mcpatlas", "toolsandbox"):
    try:
        tasks = build_tasks(Path(SUITES[name]), 999999)
    except Exception as e:
        print(name, "ERR", e)
        continue
    gold = {t.gold_name for t in tasks}
    g = max(len(gold), 1)
    row = dict(gold=len(gold),
               union_called=100*len(gold & u_called)/g, union_cat=100*len(gold & u_cat)/g,
               five_called=100*len(gold & f_called)/g, five_cat=100*len(gold & f_cat)/g)
    out[name] = row
    print(f"{name:14s} {len(gold):6d} {row['union_called']:12.1f}% {row['union_cat']:9.1f}% "
          f"{row['five_called']:11.1f}% {row['five_cat']:8.1f}%")
print("COVERAGE_JSON " + json.dumps(out))

