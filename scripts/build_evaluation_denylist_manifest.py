#!/usr/bin/env python
"""Build a provenance-bound evaluation denylist list manifest."""

from __future__ import annotations

import argparse
import json

from localagent.data.evaluation_denylist_manifest import (
    build_evaluation_denylist_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-provenance",
        action="append",
        required=True,
        metavar="PATH",
        help="frozen prompt-only suite provenance manifest (repeatable)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="canonical self-hashed denylist list manifest",
    )
    args = parser.parse_args()
    try:
        manifest = build_evaluation_denylist_manifest(
            args.suite_provenance,
            output_path=args.out,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"build failed: {error}") from error
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
