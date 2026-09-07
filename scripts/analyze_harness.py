#!/usr/bin/env python
"""Audit every evaluation suite from first principles: what would a predictor that never reads
the observation score, is the gold answer even a function of the input, and how much of it is
copyable from the prompt.

A benchmark score is unreadable without these. The project reported agentnet type_match of 74%
before measuring that always-`click` scores 73%, and mind2web step_success of 64% before finding
that 214 of 252 eval observations are other steps of tasks in the training split. Every number
in this audit is computed from the suite files the runner loads, so it is reproducible and it
moves when the data does.

Writes results/analysis/harness-audit.json and prints a table. Run on the box, where the data is.

  python scripts/analyze_harness.py
  python scripts/analyze_harness.py --suites mind2web,toolbench --train-overlap
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from openlocalagent.eval import suite as S  # noqa: E402
from openlocalagent.eval.interaction import SPECS, Floors  # noqa: E402

#: Second-person persona briefs, excluding the "You are a <role>" system prefix some suites use.
PERSONA = re.compile(r"\byou (want|should|need|don't|recently|would like|are (?!a |an ))", re.I)

#: Where each suite's posttrain split lives, for the overlap check. A suite that trains on
#: nothing (held out) is simply absent here.
TRAIN_SPLITS = {
    "mind2web": ["data/merged-v2/train.jsonl", "data/public/mind2web-train.jsonl"],
    "toolbench": ["data/public/toolbench-train.jsonl"],
    "xlam": ["data/public/xlam-train.jsonl"],
    "toolace": ["data/public/toolace-train.jsonl"],
    "androidcontrol": ["data/public/androidcontrol-train.jsonl"],
    "mobileactions": ["data/public/mobileactions-train.jsonl"],
}


def _gold_pairs(path: Path) -> set[tuple[str, str]]:
    """Every (tool name, sorted-json arguments) a training split teaches as an answer."""
    out: set[tuple[str, str]] = set()
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for m in json.loads(line).get("messages", []):
                for call in m.get("tool_calls") or []:
                    out.add((call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True)))
    return out


def _observation_hashes(path: Path, prefix: int | None = None) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = " ".join(m.get("content", "") for m in row.get("messages", [])
                            if m.get("role") != "assistant")
            out.add(hashlib.sha1((text[:prefix] if prefix else text).encode()).hexdigest())
    return out


def duplicate_ambiguity(tasks) -> dict:
    """Identical observations with different golds: rows whose answer is not a function of input."""
    groups: dict[str, list] = collections.defaultdict(list)
    for t in tasks:
        groups[hashlib.sha1(t.observation.encode()).hexdigest()].append(
            (t.gold_name, json.dumps(t.gold_arguments, sort_keys=True)))
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    return {
        "duplicate_groups": len(dups),
        "rows_in_duplicates": sum(len(v) for v in dups.values()),
        "ambiguous_groups": sum(1 for v in dups.values() if len(set(v)) > 1),
        "rows_ambiguous": sum(len(v) for v in dups.values() if len(set(v)) > 1),
    }


def train_overlap(name: str, tasks) -> dict | None:
    paths = [Path(p) for p in TRAIN_SPLITS.get(name, [])]
    if not paths:
        return None
    exact = set().union(*(_observation_hashes(p) for p in paths))
    prefix = set().union(*(_observation_hashes(p, 60) for p in paths))
    exact_rows = [t for t in tasks if hashlib.sha1(t.observation.encode()).hexdigest() in exact]
    # Same first 60 chars = same task sentence with a different step. That is episode leakage,
    # which exact matching cannot see and which is how mind2web scored 64%. It is only a leak
    # where observations open with task-specific text; on androidcontrol and mobileactions a
    # shared template makes it 100% and meaningless, so only suites tagged episode_leaked treat
    # it as one.
    prefix_rows = [t for t in tasks
                   if hashlib.sha1(t.observation[:60].encode()).hexdigest() in prefix]
    # The gold itself as a training label. mind2web's answer is an element id such as '107':
    # not derivable from the task sentence, but every one of its 252 eval golds is verbatim
    # among 4,508 training golds, and the rows the prefix test called clean scored the same
    # 58% as the rest. That is label memorisation, and it is the leak unit that matters there.
    golds = set().union(*(_gold_pairs(p) for p in paths))
    gold_rows = [t for t in tasks
                 if (t.gold_name, json.dumps(t.gold_arguments, sort_keys=True)) in golds]
    leaked = list(exact_rows)
    if name in SPECS and "episode_leaked" in SPECS[name].tags:
        leaked += gold_rows
    return {
        "exact": len(exact_rows),
        "prefix60": len(prefix_rows),
        "gold_pair_in_train": len(gold_rows),
        "leaked_hashes": sorted({hashlib.sha1(t.observation.encode()).hexdigest() for t in leaked}),
    }


def audit(name: str, with_overlap: bool) -> dict:
    path = S.SUPPLEMENTAL_SUITES.get(name) or S.SUITES[name]
    tasks = S.build_tasks(path, limit=10**9)
    n = len(tasks)
    names = collections.Counter(t.gold_name for t in tasks)
    catalogs = [len(t.tools) for t in tasks]
    record = {
        "rows": n,
        "distinct_gold_tools": len(names),
        "majority_tool": names.most_common(1)[0][0],
        "floors": Floors.measure(tasks).as_record(),
        "arguments_per_call": round(sum(len(t.gold_arguments or {}) for t in tasks) / n, 2),
        "persona_share": round(sum(1 for t in tasks if PERSONA.search(t.observation)) / n, 4),
        "catalog_mean": round(sum(catalogs) / n, 1),
        "catalog_max": max(catalogs),
        "observation_chars_mean": round(sum(len(t.observation) for t in tasks) / n),
        **duplicate_ambiguity(tasks),
        "spec_audit": SPECS[name].audit(n) if name in SPECS else [],
    }
    if with_overlap:
        record["train_overlap"] = train_overlap(name, tasks)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", help="comma-separated subset; default all ten plus tau2")
    parser.add_argument("--train-overlap", action="store_true",
                        help="also measure eval/train observation overlap (reads train splits)")
    parser.add_argument("--out", default="results/analysis/harness-audit.json")
    args = parser.parse_args()

    names = (args.suites.split(",") if args.suites else list(S.SUITES) + ["tau2"])
    report = {name: audit(name, args.train_overlap) for name in names}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.train_overlap:
        # The runner reads this to score every receipt's clean subset beside its full number.
        index = {name: {"hashes": r["train_overlap"]["leaked_hashes"],
                        "rule": "exact" + (" + gold_pair_in_train"
                                           if "episode_leaked" in SPECS[name].tags else "")}
                 for name, r in report.items()
                 if r.get("train_overlap") and r["train_overlap"]["leaked_hashes"]}
        (out.parent / "leaked-eval-rows.json").write_text(json.dumps(index, indent=1) + "\n")
        for r in report.values():
            if r.get("train_overlap"):
                r["train_overlap"]["leaked"] = len(r["train_overlap"].pop("leaked_hashes"))
    out.write_text(json.dumps(report, indent=2) + "\n")

    head = f"{'suite':<15}{'rows':>6}{'tools':>6}{'majTM%':>7}{'constSS%':>9}{'noarg%':>7}" \
           f"{'deriv%':>7}{'reach%':>7}{'dupRows':>8}{'ambig':>6}{'cat':>6}"
    print(head)
    for name, r in report.items():
        f = r["floors"]
        print(f"{name:<15}{r['rows']:>6}{r['distinct_gold_tools']:>6}"
              f"{100 * f['majority_type_match']:>7.1f}{100 * f['constant_step_success']:>9.1f}"
              f"{100 * f['no_argument_share']:>7.1f}"
              f"{(100 * f['argument_derivability']) if f['argument_derivability'] is not None else 0:>7.1f}"
              f"{100 * f['reachable_step_success']:>7.1f}{r['rows_in_duplicates']:>8}"
              f"{r['rows_ambiguous']:>6}{r['catalog_mean']:>6}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
