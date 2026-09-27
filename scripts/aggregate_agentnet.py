#!/usr/bin/env python
"""Aggregate AgentNet/OpenCUA action predictions without launching a desktop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.agentnet_results import aggregate_agentnet_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--expected-id",
        action="append",
        dest="expected_ids",
        help="expected task id; repeat once per case",
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = aggregate_agentnet_results(
        args.ground_truth,
        args.predictions,
        expected_ids=args.expected_ids,
        source_revision=args.source_revision,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "overall": receipt["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
