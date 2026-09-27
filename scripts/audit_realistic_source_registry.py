#!/usr/bin/env python3
"""Audit the canonical and supplemental realistic-agent source registries.

The canonical JSON matrix contains rows eligible for explicit train/eval policy, while the
supplemental YAML registry is catalog-only discovery.  This audit joins them without promoting
supplemental rows to training data, reports duplicate IDs and metadata conflicts, and binds both
source files by SHA-256 so later acquisitions can be compared against the same inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from localagent.data.public_eval_matrix import load_matrix


KIND = "localagent_realistic_source_registry_audit"
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


def _self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _load_supplemental(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read supplemental registry {path}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("kind") != "localagent_realistic_agent_supplemental_catalog":
        raise ValueError("supplemental registry has an unexpected kind")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("supplemental registry entries must be a non-empty list")
    return dict(raw)


def _row(catalog: str, item: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the source identity and policy fields needed for an acquisition review."""

    source_url = item.get("source_url")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise ValueError(f"{catalog} row {item.get('id')!r} has an invalid source_url")
    row = {
        "catalog": catalog,
        "id": item.get("id"),
        "family": item.get("family"),
        "name": item.get("name"),
        "source_url": source_url,
        "paper_url": item.get("paper_url"),
        "source_revision": item.get("source_revision"),
        "license": item.get("license"),
        "train_policy": item.get("train_policy", "catalog_only"),
        "access_status": item.get("access_status", "catalog_only"),
        "local_status": item.get("local_status", "catalog_only"),
        "split_policy": item.get("split_policy", item.get("split_rule")),
        "runtime": item.get("runtime"),
        "webgpu_projection": item.get("webgpu_projection"),
    }
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError(f"{catalog} row has no id")
    if not isinstance(row["family"], str) or not row["family"]:
        raise ValueError(f"{catalog} row {row['id']!r} has no family")
    return row


def audit(matrix_path: Path, supplemental_path: Path) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    supplemental = _load_supplemental(supplemental_path)
    canonical_rows = [_row("canonical", item) for item in matrix["entries"]]
    supplemental_rows = [_row("supplemental", item) for item in supplemental["entries"]]
    all_rows = canonical_rows + supplemental_rows

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_id[row["id"]].append(row)
    overlap_ids = sorted(key for key, rows in by_id.items() if len(rows) > 1)
    conflicts: list[dict[str, Any]] = []
    for entry_id in overlap_ids:
        rows = by_id[entry_id]
        for field in ("source_url", "family", "source_revision", "license"):
            values = sorted({str(row.get(field)) for row in rows})
            if len(values) > 1:
                conflicts.append({"id": entry_id, "field": field, "values": values})

    payload: dict[str, Any] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "canonical_matrix": {
                "path": str(matrix_path),
                "bytes": matrix_path.stat().st_size,
                "sha256": _sha256(matrix_path),
                "entries": len(canonical_rows),
            },
            "supplemental_registry": {
                "path": str(supplemental_path),
                "bytes": supplemental_path.stat().st_size,
                "sha256": _sha256(supplemental_path),
                "entries": len(supplemental_rows),
            },
        },
        "counts": {
            "canonical_entries": len(canonical_rows),
            "supplemental_entries": len(supplemental_rows),
            "unique_source_ids": len(by_id),
            "overlapping_ids": len(overlap_ids),
            "metadata_conflicts": len(conflicts),
            "families_all_rows": dict(sorted(Counter(row["family"] for row in all_rows).items())),
            "families_unique_ids": dict(
                sorted(Counter(by_id[key][0]["family"] for key in by_id).items())
            ),
            "canonical_train_rows": sum(row["train_policy"] == "train" for row in canonical_rows),
            "supplemental_catalog_only_rows": sum(
                row["train_policy"] == "catalog_only" for row in supplemental_rows
            ),
        },
        "overlap_ids": overlap_ids,
        "metadata_conflicts": conflicts,
        "train_eligible": [
            {
                "id": row["id"],
                "source_url": row["source_url"],
                "paper_url": row["paper_url"],
                "source_revision": row["source_revision"],
                "license": row["license"],
            }
            for row in canonical_rows
            if row["train_policy"] == "train"
        ],
        "sources": sorted(all_rows, key=lambda row: (row["id"], row["catalog"])),
        "claim_boundary": (
            "This is a source-linked inventory audit, not a dataset acquisition or benchmark run. "
            "The canonical rows retain explicit train/eval policy; every supplemental row remains "
            "catalog_only. No task prompts, screenshots, credentials, emulator/VM images, MCP "
            "service state, verifier traces, or live side effects are included. A train-policy row "
            "still requires a separate license-reviewed, revision/hash-bound acquisition receipt "
            "and source-specific adapter."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/data/realistic-agent-public-eval-matrix.v1.json"),
    )
    parser.add_argument(
        "--supplemental",
        type=Path,
        default=Path("configs/data/realistic-agent-eval.supplemental.yaml"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.matrix, args.supplemental)
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
