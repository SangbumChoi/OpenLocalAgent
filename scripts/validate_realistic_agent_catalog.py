#!/usr/bin/env python
"""Validate the source-linked realistic-agent catalog and print its audit fingerprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.realistic_catalog import eval_entries, load_catalog, train_entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog",
        nargs="?",
        default="configs/data/realistic-agent-eval.catalog.yaml",
        help="YAML catalog path",
    )
    args = parser.parse_args()
    catalog, fingerprint = load_catalog(Path(args.catalog))
    train = train_entries(catalog)
    evaluation = eval_entries(catalog)
    summary = {
        "kind": catalog["kind"],
        "schema_version": catalog["schema_version"],
        "catalog_sha256": fingerprint,
        "entries": len(catalog["entries"]),
        "train_entries": [row["id"] for row in train],
        "evaluation_or_restricted_entries": [row["id"] for row in evaluation],
        "policy": "Only train_entries may be passed to an acquisition/training config; all other rows are holdouts or runtime evaluators.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
