#!/usr/bin/env python3
"""Profile the public CUA-Gym task table without executing task artifacts.

The CUA-Gym Hub release publishes a small metadata parquet alongside a much larger archive of
Python/shell/application setup and reward files.  This adapter consumes only a JSON export from
the public dataset-server rows endpoint (or an equivalent ``{"rows": [...]}`` payload), records
coverage and split/licensing provenance, and deliberately omits instruction text and executable
artifact contents from the receipt.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


DATASET = "xlangai/CUA-Gym"
SOURCE_URL = "https://huggingface.co/datasets/xlangai/CUA-Gym"
SOURCE_REVISION = "3c021d0"
LICENSE = "CC-BY-4.0"
REQUIRED_FIELDS = {
    "id",
    "instruction",
    "app_type",
    "app_family",
    "platform",
    "difficulty",
    "setup_kind",
    "num_setup_steps",
    "num_setup_files",
    "has_ground_truth",
    "setup_files",
    "archive_path",
    "archive_member",
    "task_json_member",
    "reward_member",
    "setup_file_members",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        value = row.get(field)
        key = "<null>" if value is None else str(value)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read CUA-Gym metadata export: {path}") from error
    raw_rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("CUA-Gym metadata export must contain a non-empty rows list")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = raw.get("row") if isinstance(raw, dict) and isinstance(raw.get("row"), dict) else raw
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"row {index} missing required fields: {missing}")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError(f"row {index} has an invalid id")
        if not isinstance(row["instruction"], str):
            raise ValueError(f"row {index} has an invalid instruction")
        if not isinstance(row["num_setup_steps"], int) or row["num_setup_steps"] < 1:
            raise ValueError(f"row {index} has an invalid num_setup_steps")
        if row["num_setup_files"] != 1:
            raise ValueError(f"row {index} has unexpected num_setup_files")
        if not isinstance(row["has_ground_truth"], bool):
            raise ValueError(f"row {index} has an invalid has_ground_truth value")
        for field in ("setup_files", "setup_file_members"):
            if not isinstance(row[field], list) or not all(isinstance(item, str) for item in row[field]):
                raise ValueError(f"row {index} has invalid {field}")
        rows.append(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("CUA-Gym task ids must be unique")
    return rows


def profile(path: Path, *, revision: str = SOURCE_REVISION) -> dict[str, Any]:
    """Return a deterministic, prompt-redacted CUA-Gym coverage receipt."""

    rows = _load_rows(path)
    setup_steps = [int(row["num_setup_steps"]) for row in rows]
    archive_paths = _counter(rows, "archive_path")
    if len(archive_paths) != 1:
        raise ValueError("CUA-Gym rows must reference one pinned artifact archive")
    payload: dict[str, Any] = {
        "kind": "localagent_cua_gym_metadata_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": revision,
        "license": LICENSE,
        "source": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "format": "dataset_server_rows_json",
            "instruction_text_retained": False,
            "task_artifacts_downloaded": False,
            "reward_code_executed": False,
        },
        "split": {
            "name": "train",
            "rows": len(rows),
            "official_eval_split_present": False,
            "policy": "metadata_inventory_only_until_a_source_specific_train_holdout_and_artifact_review_are_frozen",
        },
        "coverage": {
            "unique_task_ids": len(rows),
            "app_types": len(_counter(rows, "app_type")),
            "app_type_counts": _counter(rows, "app_type"),
            "app_family_counts": _counter(rows, "app_family"),
            "platform_counts": _counter(rows, "platform"),
            "difficulty_counts": _counter(rows, "difficulty"),
            "setup_kind_counts": _counter(rows, "setup_kind"),
            "ground_truth_counts": _counter(rows, "has_ground_truth"),
            "empty_instruction_rows": sum(not row["instruction"] for row in rows),
            "setup_steps": {
                "min": min(setup_steps),
                "max": max(setup_steps),
                "mean": statistics.mean(setup_steps),
                "median": statistics.median(setup_steps),
            },
            "archive_paths": archive_paths,
            "artifact_kinds": _counter(rows, "setup_kind"),
        },
        "claim_boundary": (
            "Public CUA-Gym metadata inventory only. The receipt excludes instruction text and all "
            "setup/reward artifacts; it is not an RLVR training run, task-success score, native "
            "desktop/browser evaluation, or permission to execute unreviewed artifact code."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")
    payload = profile(args.input, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
