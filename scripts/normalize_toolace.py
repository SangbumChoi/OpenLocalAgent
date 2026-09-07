#!/usr/bin/env python3
"""Normalize a byte-pinned public ToolACE snapshot into Conversation JSONL splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openlocalagent.data.adapters.toolace import normalize_toolace_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-eval", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--projection",
        choices=("first_action", "multiturn", "action_history"),
        default="first_action",
        help="retain only the first action, preserve full history, or keep action-relevant history",
    )
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    manifest = normalize_toolace_snapshot(
        args.input,
        output_train=args.output_train,
        output_eval=args.output_eval,
        manifest_path=args.manifest,
        expected_bytes=args.expected_bytes,
        expected_sha256=args.expected_sha256,
        projection=args.projection,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
