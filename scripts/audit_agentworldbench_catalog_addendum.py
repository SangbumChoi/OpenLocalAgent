#!/usr/bin/env python3
"""Validate and hash-bind the current AgentWorldBench catalog addendum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.realistic_catalog import load_catalog


DATASET = "Qwen/AgentWorldBench"
REVISION = "6b8d28437042434dcdd168434227ca0de408c5ba"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def audit(catalog_path: Path, projection_receipt: Path) -> dict[str, Any]:
    catalog, catalog_sha256 = load_catalog(catalog_path)
    rows = [row for row in catalog["entries"] if row["id"] == "agentworldbench"]
    if len(rows) != 1:
        raise ValueError("addendum must contain exactly one agentworldbench row")
    row = rows[0]
    if row["train_policy"] != "eval_only" or row["source_revision"] != REVISION:
        raise ValueError("AgentWorldBench addendum is not pinned eval-only")
    projection = json.loads(projection_receipt.read_text(encoding="utf-8"))
    if projection.get("kind") != "localagent_agentworldbench_text_projection_eval":
        raise ValueError("unexpected AgentWorldBench projection receipt kind")
    if projection["manifest"]["source_revision"] != REVISION:
        raise ValueError("projection revision does not match catalog addendum")
    body: dict[str, Any] = {
        "kind": "localagent_agentworldbench_catalog_addendum_audit",
        "schema_version": 1,
        "catalog": {
            "path": str(catalog_path),
            "sha256": catalog_sha256,
            "entries": len(catalog["entries"]),
            "dataset": DATASET,
            "source_revision": REVISION,
            "train_policy": row["train_policy"],
        },
        "projection_receipt": {
            "path": str(projection_receipt),
            "sha256": projection["receipt_self_sha256"],
            "rows": projection["protocol"]["rows"],
            "domains": projection["protocol"]["domains"],
        },
        "decision": {
            "training_admission": False,
            "official_native_score": False,
            "catalog_discovery": True,
        },
        "claim_boundary": (
            "This addendum makes AgentWorldBench discoverable as a pinned eval-only source. "
            "It does not authorize training on test rows or convert the teacher-forced projection "
            "into an official judge/native benchmark score."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/data/realistic-agent-eval.agentworldbench.yaml"))
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = audit(args.catalog, args.projection_receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
