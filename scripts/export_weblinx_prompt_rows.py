#!/usr/bin/env python
"""Export private WebLINX prompt-only rows from immutable local chat and split files.

Example:
  python scripts/export_weblinx_prompt_rows.py \
    --revision be2e19d624febb57173e98772c1312d041a6d3b1 \
    --split test_web \
    --chat data/private/weblinx/test_web.json.gz 2187263 <sha256> \
    --splits data/private/weblinx/splits.json 38210 <sha256> \
    --output data/private/weblinx-test-web-prompts.jsonl \
    --audit data/private/weblinx-test-web-prompts.audit.json

The command excludes whole demonstrations under deterministic credential/PII rules.  It never
downloads data and does not implement official WebLINX scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.weblinx_prompts import (
    DEFAULT_MAX_CHAT_SOURCE_BYTES,
    DEFAULT_MAX_DECOMPRESSED_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MAX_RECORD_CHARS,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_SPLITS_BYTES,
    WebLINXSource,
    export_weblinx_prompt_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        required=True,
        help="immutable lowercase 40-character upstream dataset commit",
    )
    parser.add_argument("--split", required=True, help="split key from the pinned splits.json")
    parser.add_argument(
        "--chat",
        nargs=3,
        metavar=("PATH", "BYTES", "SHA256"),
        required=True,
        help="local compact chat JSON/JSON.GZ plus immutable byte size and SHA-256",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        metavar=("PATH", "BYTES", "SHA256"),
        required=True,
        help="local splits.json plus immutable byte size and SHA-256",
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
    parser.add_argument(
        "--max-chat-source-bytes",
        type=int,
        default=DEFAULT_MAX_CHAT_SOURCE_BYTES,
    )
    parser.add_argument("--max-splits-bytes", type=int, default=DEFAULT_MAX_SPLITS_BYTES)
    parser.add_argument(
        "--max-decompressed-bytes",
        type=int,
        default=DEFAULT_MAX_DECOMPRESSED_BYTES,
    )
    parser.add_argument("--max-record-chars", type=int, default=DEFAULT_MAX_RECORD_CHARS)
    parser.add_argument("--max-prompt-bytes", type=int, default=DEFAULT_MAX_PROMPT_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    return parser.parse_args()


def _source(arguments: list[str]) -> WebLINXSource:
    path, expected_bytes, sha256 = arguments
    return WebLINXSource(path=Path(path), bytes=int(expected_bytes), sha256=sha256)


def main() -> None:
    args = parse_args()
    try:
        audit = export_weblinx_prompt_rows(
            _source(args.chat),
            _source(args.splits),
            args.output,
            revision=args.revision,
            split=args.split,
            audit_path=args.audit,
            max_chat_source_bytes=args.max_chat_source_bytes,
            max_splits_bytes=args.max_splits_bytes,
            max_decompressed_bytes=args.max_decompressed_bytes,
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
