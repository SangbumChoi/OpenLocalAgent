#!/usr/bin/env python
"""Normalize decoder-produced mobile rows into the audited localagent_v1 JSONL format.

This script does not parse TFRecords or download data.  An upstream-specific decoder must first
emit the intermediate row documented in ``localagent.data.realistic_adapters``.  The output can
then be consumed by the existing hash-pinned public-agent ingestion config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from localagent.data.realistic_adapters import normalize_mobile_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("androidcontrol", "android_in_the_wild"), required=True)
    parser.add_argument("--revision", required=True, help="upstream revision bound to this export")
    parser.add_argument("--input", required=True, type=Path, help="decoder-produced JSONL")
    parser.add_argument("--output", required=True, type=Path, help="localagent_v1 JSONL")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--output must differ from --input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                normalized = normalize_mobile_row(
                    raw,
                    family=args.family,
                    source_revision=args.revision,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise SystemExit(f"line {line_number}: {error}") from error
            encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            destination.write(encoded + "\n")
            digest.update((encoded + "\n").encode("utf-8"))
            count += 1
    print(json.dumps({"records": count, "output_sha256": digest.hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
