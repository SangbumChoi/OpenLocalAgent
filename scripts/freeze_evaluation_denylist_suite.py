#!/usr/bin/env python
"""Freeze one contract-bound prompt-only evaluation denylist suite."""

from __future__ import annotations

import argparse
import json

from localagent.data.evaluation_denylist_suite import (
    freeze_evaluation_denylist_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", help="strict JSON suite contract")
    parser.add_argument("--output", required=True, help="canonical prompt-only JSONL output")
    parser.add_argument("--manifest", required=True, help="self-hashed provenance manifest")
    args = parser.parse_args()
    try:
        manifest = freeze_evaluation_denylist_suite(
            args.contract,
            output_path=args.output,
            manifest_path=args.manifest,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"freeze failed: {error}") from error
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
