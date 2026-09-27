#!/usr/bin/env python
"""Score every already-evaluated arm on one added suite, merging into its existing report.

Each report records the model spec it was produced from, so the arms are replayed from their own
receipts rather than from a hand-maintained list that can drift out of step with what was run.

  python scripts/score_toolbench.py --suite toolbench
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPORTS = Path("runs/evalsuite")
# Reports that are baselines or aggregates rather than a model, so there is nothing to re-score.
NOT_MODELS = {"chance-baseline", "majority-baseline", "dispatch"}


def model_spec(report: dict) -> str | None:
    """Rebuild the --model argument this report was produced from."""
    kind, location = report.get("kind"), report.get("location")
    if not location:
        return None
    if kind in ("hf", "lora", "catalog"):
        return f"{kind}:{location}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="toolbench")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--status", default="explog/TOOLBENCH_STATUS.txt")
    # The two boxes do not run the same model kinds: only one has a working transformers, so the
    # HF-backed arms and the local checkpoints are scored where each can actually load.
    ap.add_argument("--kinds", default="hf,lora,catalog")
    args = ap.parse_args()

    status = Path(args.status)
    status.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(REPORTS.glob("*.json")):
        tag = path.stem
        if tag in NOT_MODELS:
            continue
        report = json.loads(path.read_text())
        if args.suite in report.get("suites", {}):
            continue
        if report.get("kind") not in set(args.kinds.split(",")):
            continue
        spec = model_spec(report)
        if spec is None:
            with status.open("a") as handle:
                handle.write(f"{tag} skipped (no replayable spec)\n")
            continue
        started = time.time()
        result = subprocess.run(
            [sys.executable, "scripts/eval_suite.py", "--model", spec, "--rows", str(args.rows),
             "--device", args.device, "--suites", args.suite, "--out", str(path)],
            capture_output=True, text=True)
        Path(f"explog/tb_{tag}.log").write_text(result.stdout + result.stderr)
        with status.open("a") as handle:
            handle.write(f"{tag} rc={result.returncode} secs={time.time() - started:.0f}\n")
        print(f"{tag} rc={result.returncode}", flush=True)
    with status.open("a") as handle:
        handle.write("TOOLBENCH_DONE\n")


if __name__ == "__main__":
    main()
