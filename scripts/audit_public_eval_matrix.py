#!/usr/bin/env python3
"""Audit the current public realistic-agent matrix without downloading benchmark payloads.

The matrix is a research inventory, not an authorization to acquire task text.  This command
validates every source-linked row, binds the exact matrix bytes, and records family/policy/status
counts so later native receipts can be joined without silently changing the source inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from localagent.data.public_eval_matrix import load_matrix

KIND = "localagent_public_realistic_eval_matrix_audit"
SCHEMA_VERSION = 1


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hash(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def audit(path: Path) -> dict[str, Any]:
    matrix = load_matrix(path)
    entries = matrix["entries"]
    payload: dict[str, Any] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "matrix": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "canonical_sha256": hashlib.sha256(_canonical(matrix)).hexdigest(),
            "entries": len(entries),
        },
        "counts": {
            "families": dict(sorted(Counter(row["family"] for row in entries).items())),
            "train_policy": dict(sorted(Counter(row["train_policy"] for row in entries).items())),
            "local_status": dict(sorted(Counter(row["local_status"] for row in entries).items())),
        },
        "train_eligible": [
            {
                "id": row["id"],
                "source_url": row["source_url"],
                "paper_url": row["paper_url"],
                "source_revision": row.get("source_revision"),
                "license": row["license"],
            }
            for row in entries
            if row["train_policy"] == "train"
        ],
        "sources": [
            {
                "id": row["id"],
                "family": row["family"],
                "source_url": row["source_url"],
                "paper_url": row["paper_url"],
                "source_revision": row.get("source_revision"),
                "train_policy": row["train_policy"],
                "local_status": row["local_status"],
                "primary_metric": row["primary_metric"],
                "split_rule": row["split_rule"],
            }
            for row in entries
        ],
        "claim_boundary": (
            "Source-linked metadata and local schema validation only. No benchmark task text, "
            "screenshots, credentials, emulator/VM assets, MCP service state, verifier output, "
            "or live external side effect was downloaded or executed. A train-policy row still "
            "requires a separate byte/hash acquisition receipt and source-specific adapter; "
            "runtime rows require release-matched native execution before scoring."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix",
        nargs="?",
        type=Path,
        default=Path("configs/data/realistic-agent-public-eval-matrix.v1.json"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.matrix)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite report: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
