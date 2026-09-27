#!/usr/bin/env python
"""Verify a completed parent-anchored SFT run and publish its integrity receipt."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from localagent.eval.sft_production_receipt import (
    verify_sft_production_receipt_against_artifacts,
    verify_sft_production_run,
    write_sft_production_receipt,
)

ROOT_ARGUMENTS = (
    "config-file-sha256",
    "config-canonical-sha256",
    "model-config-file-sha256",
    "model-config-canonical-sha256",
    "tokenizer-sha256",
    "data-sha256",
    "parent-checkpoint-sha256",
    "budget-plan-file-sha256",
    "budget-plan-self-sha256",
    "preflight-file-sha256",
    "preflight-self-sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="exact production SFT YAML config",
    )
    parser.add_argument(
        "--budget-plan",
        type=Path,
        required=True,
        help="canonical sealed stage-budget plan",
    )
    parser.add_argument(
        "--preflight",
        "--passed-preflight",
        dest="preflight",
        type=Path,
        required=True,
        help="canonical passed SFT execution-preflight receipt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new canonical production receipt path (never overwritten)",
    )
    for name in ROOT_ARGUMENTS:
        parser.add_argument(
            f"--expected-{name}",
            required=True,
            metavar="SHA256",
            help=f"independent expected {name.replace('-', ' ')} root",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_roots = {
        name.replace("-", "_"): getattr(args, f"expected_{name.replace('-', '_')}")
        for name in ROOT_ARGUMENTS
    }
    receipt = verify_sft_production_run(
        args.config,
        args.budget_plan,
        args.preflight,
        expected_roots=expected_roots,
    )
    write_sft_production_receipt(args.output, receipt)
    payload = args.output.read_bytes()
    receipt_file_sha256 = hashlib.sha256(payload).hexdigest()
    verify_sft_production_receipt_against_artifacts(
        args.output,
        expected_receipt_file_sha256=receipt_file_sha256,
    )
    print(f"verified production SFT run: {receipt['artifacts']['run_directory']['path']}")
    print(f"canonical receipt: {args.output}")
    print(f"receipt file SHA-256: {receipt_file_sha256}")
    print(f"receipt self SHA-256: {receipt['receipt_self_sha256']}")
    print("scope: artifact integrity and training accounting only; no quality/retention claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
