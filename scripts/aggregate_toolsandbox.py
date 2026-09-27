#!/usr/bin/env python
"""Aggregate an Apple ToolSandbox result_summary.json file without executing tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.toolsandbox import aggregate_toolsandbox_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-summary", required=True, type=Path)
    parser.add_argument(
        "--expected-scenario",
        action="append",
        dest="expected_scenarios",
        help="expected scenario name; repeat once per scenario",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = aggregate_toolsandbox_results(
        args.result_summary,
        expected_scenarios=args.expected_scenarios,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "overall": receipt["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
