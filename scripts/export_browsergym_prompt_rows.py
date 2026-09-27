#!/usr/bin/env python
"""Validate an offline BrowserGym reset capture and export prompt-only rows.

Production additionally requires the controlled producer receipt and both artifacts' identities
to have been frozen in the paper policy. It intentionally fails closed while those pins are unset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.browsergym_prompts import (
    BrowserGymPromptLimits,
    export_browsergym_prompt_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="offline reset-capture JSONL")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="controlled capture-producer receipt (required in production mode)",
    )
    parser.add_argument("--capture-bytes", type=int, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True, help="two-field prompt JSONL output")
    parser.add_argument("--audit", type=Path, required=True, help="canonical export audit JSON")
    parser.add_argument(
        "--mode",
        choices=("production", "fixture"),
        default="production",
        help=(
            "production enforces the pinned plan and separately frozen controlled-capture hash"
        ),
    )
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=BrowserGymPromptLimits.max_capture_bytes,
    )
    parser.add_argument(
        "--max-line-bytes",
        type=int,
        default=BrowserGymPromptLimits.max_line_bytes,
    )
    parser.add_argument(
        "--max-prompt-bytes",
        type=int,
        default=BrowserGymPromptLimits.max_prompt_bytes,
    )
    parser.add_argument(
        "--max-capture-rows",
        type=int,
        default=BrowserGymPromptLimits.max_capture_rows,
    )
    args = parser.parse_args()
    if args.mode == "production" and args.receipt is None:
        parser.error("--receipt is required in production mode")
    return args


def main() -> None:
    args = parse_args()
    limits = BrowserGymPromptLimits(
        max_capture_bytes=args.max_capture_bytes,
        max_line_bytes=args.max_line_bytes,
        max_prompt_bytes=args.max_prompt_bytes,
        max_capture_rows=args.max_capture_rows,
    )
    try:
        audit = export_browsergym_prompt_rows(
            args.capture,
            args.out,
            args.audit,
            expected_capture_bytes=args.capture_bytes,
            expected_capture_sha256=args.capture_sha256,
            receipt_path=args.receipt,
            production=args.mode == "production",
            limits=limits,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"BrowserGym prompt export failed: {error}") from error
    print(json.dumps(audit["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
