#!/usr/bin/env python
"""Report which realistic-agent catalog rows are locally executable.

This command performs read-only dependency probes and never downloads datasets, starts an
emulator, or claims an official benchmark score.  Use ``--strict`` in CI to fail when any catalog
row is blocked.
"""

from __future__ import annotations

import argparse
import sys

from localagent.eval.realistic_preflight import json_report, preflight_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog",
        nargs="?",
        default="configs/data/realistic-agent-eval.catalog.yaml",
        help="source-linked realistic-agent catalog",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when any catalog row is blocked",
    )
    args = parser.parse_args()
    try:
        report = preflight_catalog(args.catalog)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"realistic-agent preflight failed: {error}") from error
    sys.stdout.write(json_report(args.catalog))
    return 1 if args.strict and report["blocked_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
