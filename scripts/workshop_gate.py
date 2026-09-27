#!/usr/bin/env python
"""Build the fail-closed workshop/publication readiness report."""

from __future__ import annotations

import argparse
import json
import sys

from localagent.eval.workshop_gate import build_workshop_gate, write_workshop_gate


def _receipt(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("native receipt must be BENCHMARK_ID=PATH")
    benchmark_id, path = value.split("=", 1)
    if not benchmark_id or not path:
        raise argparse.ArgumentTypeError("native receipt must be BENCHMARK_ID=PATH")
    return benchmark_id, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/data/realistic-agent-eval.catalog.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--native-receipt",
        action="append",
        type=_receipt,
        default=[],
        metavar="BENCHMARK_ID=PATH",
    )
    parser.add_argument("--webgpu-receipt")
    parser.add_argument("--weight-report", action="append", default=[])
    parser.add_argument("--public-artifact-manifest")
    parser.add_argument("--rl-preflight-receipt")
    parser.add_argument(
        "--current-checkpoint",
        help="checkpoint path whose SHA-256 must be bound by the public manifest",
    )
    parser.add_argument("--output")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any requirement is blocked")
    args = parser.parse_args()
    report = build_workshop_gate(
        args.catalog,
        repo_root=args.repo_root,
        native_receipts=dict(args.native_receipt),
        webgpu_receipt=args.webgpu_receipt,
        weight_reports=args.weight_report,
        public_artifact_manifest=args.public_artifact_manifest,
        rl_preflight_receipt=args.rl_preflight_receipt,
        current_checkpoint=args.current_checkpoint,
    )
    if args.output:
        write_workshop_gate(report, args.output)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
