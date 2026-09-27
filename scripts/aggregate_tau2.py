#!/usr/bin/env python
"""Aggregate official tau2-bench Results JSON without running the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.tau2 import aggregate_tau2_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="results.json or a tau2 result directory",
    )
    parser.add_argument(
        "--expected-case",
        action="append",
        dest="expected_cases",
        help="expected domain/task_id@trial case; repeat once per case",
    )
    parser.add_argument("--expected-trials", type=int)
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = aggregate_tau2_results(
        args.results,
        expected_cases=args.expected_cases,
        expected_trials=args.expected_trials,
        source_revision=args.source_revision,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "overall": receipt["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
