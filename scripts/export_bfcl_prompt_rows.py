#!/usr/bin/env python
"""Export identity-bound BFCL-v4 question/tool prompts for corpus decontamination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.bfcl_prompts import BFCLPromptLimits, export_bfcl_prompt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_manifest",
        type=Path,
        help="versioned manifest declaring the BFCL revision, categories, and source identities",
    )
    parser.add_argument("--out", type=Path, required=True, help="two-field prompt JSONL output")
    parser.add_argument("--audit", type=Path, required=True, help="canonical export audit JSON")
    parser.add_argument(
        "--max-manifest-bytes",
        type=int,
        default=BFCLPromptLimits.max_manifest_bytes,
    )
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        default=BFCLPromptLimits.max_source_bytes,
    )
    parser.add_argument(
        "--max-total-source-bytes",
        type=int,
        default=BFCLPromptLimits.max_total_source_bytes,
    )
    parser.add_argument(
        "--max-line-bytes",
        type=int,
        default=BFCLPromptLimits.max_line_bytes,
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=BFCLPromptLimits.max_source_rows,
    )
    parser.add_argument(
        "--max-output-rows",
        type=int,
        default=BFCLPromptLimits.max_output_rows,
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=BFCLPromptLimits.max_sources,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limits = BFCLPromptLimits(
        max_manifest_bytes=args.max_manifest_bytes,
        max_source_bytes=args.max_source_bytes,
        max_total_source_bytes=args.max_total_source_bytes,
        max_line_bytes=args.max_line_bytes,
        max_source_rows=args.max_source_rows,
        max_output_rows=args.max_output_rows,
        max_sources=args.max_sources,
    )
    try:
        audit = export_bfcl_prompt_rows(
            args.source_manifest,
            args.out,
            args.audit,
            limits=limits,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"BFCL prompt export failed: {error}") from error
    print(json.dumps(audit["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
