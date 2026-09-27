#!/usr/bin/env python
"""Freeze the active BrowserGym capture environment into a new canonical manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.browsergym_runtime_manifest import (
    freeze_active_environment_manifest,
)


def parse_args() -> argparse.Namespace:
    """Parse the no-clobber manifest output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new canonical environment manifest path; existing files are never overwritten",
    )
    return parser.parse_args()


def main() -> None:
    """Build the active identity, publish it, and print its compact signed identity."""

    args = parse_args()
    try:
        manifest = freeze_active_environment_manifest(args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"BrowserGym runtime manifest freeze failed: {error}") from error
    print(
        json.dumps(
            {
                "distributions": len(manifest["installed_distributions"]),
                "manifest_self_sha256": manifest["manifest_self_sha256"],
                "playwright_driver_sha256": manifest["playwright_driver"]["content"][
                    "sha256"
                ],
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
