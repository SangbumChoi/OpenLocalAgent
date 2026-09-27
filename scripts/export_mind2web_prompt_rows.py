#!/usr/bin/env python
"""Export private Mind2Web prompt-only rows from immutable local JSON shards.

Example:
  python scripts/export_mind2web_prompt_rows.py \
    --revision 17ece8eb89862368edc0cc806acee6fca5163474 \
    --split cross_domain+cross_task+cross_website \
    --archive data/private/mind2web/test.zip 567745122 \
      8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89 \
    --member-source test_domain/test_domain_0.json \
      data/private/mind2web/test_domain/test_domain_0.json <bytes> <sha256> \
    --member-source test_domain/test_domain_1.json \
      data/private/mind2web/test_domain/test_domain_1.json <bytes> <sha256> \
    # ...all 15 exact protected archive members... \
    --ranker-config configs/data/mind2web-dom-lexical-v1.json \
    --output data/private/mind2web-prompts.jsonl \
    --audit data/private/mind2web-prompts.audit.json

The output contains protected benchmark prompts and must remain private.  This command never
downloads data and does not implement official Mind2Web scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.mind2web_prompts import (
    DEFAULT_MAX_ARCHIVE_BYTES,
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MAX_RECORD_CHARS,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_SOURCE_BYTES,
    DEFAULT_MAX_SOURCES,
    DEFAULT_MAX_TOTAL_SOURCE_BYTES,
    Mind2WebArchive,
    Mind2WebSource,
    export_mind2web_prompt_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        required=True,
        help="immutable lowercase 40-character upstream dataset commit",
    )
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "declared private split; the paper bundle uses "
            "cross_domain+cross_task+cross_website"
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("PATH", "BYTES", "SHA256"),
        help="fixture shard plus its caller-supplied immutable byte size and SHA-256",
    )
    parser.add_argument(
        "--member-source",
        action="append",
        nargs=4,
        metavar=("MEMBER", "PATH", "BYTES", "SHA256"),
        help=(
            "production extracted shard: exact protected ZIP member path, local path, "
            "byte size, and SHA-256"
        ),
    )
    parser.add_argument(
        "--archive",
        nargs=3,
        metavar=("PATH", "BYTES", "SHA256"),
        help="protected test.zip plus its pinned byte size and SHA-256",
    )
    parser.add_argument(
        "--ranker-config",
        type=Path,
        help=(
            "explicit canonical self-hashed DOM-ranker config; required for the exact "
            "production revision/split and forbidden for fixture exports"
        ),
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        required=True,
        help="private canonical JSONL output",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="optional separate canonical adapter-audit JSON",
    )
    parser.add_argument("--max-archive-bytes", type=int, default=DEFAULT_MAX_ARCHIVE_BYTES)
    parser.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    parser.add_argument(
        "--max-total-source-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_SOURCE_BYTES,
    )
    parser.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES)
    parser.add_argument(
        "--max-compression-ratio",
        type=int,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
    )
    parser.add_argument("--max-record-chars", type=int, default=DEFAULT_MAX_RECORD_CHARS)
    parser.add_argument(
        "--max-prompt-bytes",
        type=int,
        default=None,
        help=(
            f"fixture prompt cap (default {DEFAULT_MAX_PROMPT_BYTES}); production is pinned "
            "by --ranker-config and accepts only that exact value"
        ),
    )
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sources = [
            Mind2WebSource(
                path=Path(path),
                bytes=int(expected_bytes),
                sha256=sha256,
            )
            for path, expected_bytes, sha256 in (args.source or [])
        ]
        sources.extend(
            Mind2WebSource(
                path=Path(path),
                bytes=int(expected_bytes),
                sha256=sha256,
                archive_member=member,
            )
            for member, path, expected_bytes, sha256 in (args.member_source or [])
        )
        archive = (
            Mind2WebArchive(
                path=Path(args.archive[0]),
                bytes=int(args.archive[1]),
                sha256=args.archive[2],
            )
            if args.archive is not None
            else None
        )
        audit = export_mind2web_prompt_rows(
            sources,
            args.output,
            revision=args.revision,
            split=args.split,
            archive=archive,
            audit_path=args.audit,
            ranker_config_path=args.ranker_config,
            max_archive_bytes=args.max_archive_bytes,
            max_source_bytes=args.max_source_bytes,
            max_total_source_bytes=args.max_total_source_bytes,
            max_sources=args.max_sources,
            max_compression_ratio=args.max_compression_ratio,
            max_record_chars=args.max_record_chars,
            max_prompt_bytes=args.max_prompt_bytes,
            max_output_bytes=args.max_output_bytes,
            max_rows=args.max_rows,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
