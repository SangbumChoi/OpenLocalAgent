#!/usr/bin/env python
"""Aggregate normalized BrowserGym/Gymnasium episode JSONL without launching a browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.browsergym import aggregate_browsergym_episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument(
        "--expected-case",
        action="append",
        dest="expected_cases",
        help="expected task_id@seed case; repeat once per case",
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--miniwob-revision")
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = aggregate_browsergym_episodes(
        args.episodes,
        expected_cases=args.expected_cases,
        source_revision=args.source_revision,
        miniwob_revision=args.miniwob_revision,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "overall": receipt["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
