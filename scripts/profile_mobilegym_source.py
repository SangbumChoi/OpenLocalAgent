#!/usr/bin/env python3
"""Profile MobileGym's public source split manifests without retaining task text.

MobileGym publishes task definitions in its source repository and separates benchmark content
under a non-commercial data license.  This profiler extracts only the official split lists from a
hash-pinned source archive, records byte/hash and family-count metadata, and rejects malformed or
overlapping train/test manifests.  It does not launch the simulator, copy task prompts, or make a
native model-score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY = "Purewhiter/mobilegym"
REPOSITORY_URL = "https://github.com/Purewhiter/mobilegym"
REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
ARCHIVE_URL = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{REVISION}"
SPLITS = ("train", "test", "payment", "high_risk")


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _split_payload(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"MobileGym {name}.txt is not UTF-8") from error
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"MobileGym {name}.txt contains duplicate task IDs")
    if any("." not in row for row in rows):
        raise ValueError(f"MobileGym {name}.txt contains an unscoped task ID")
    families = Counter(row.split(".", 1)[0] for row in rows)
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tasks": len(rows),
        "families": dict(sorted(families.items())),
    }


def profile(archive: Path, *, revision: str = REVISION) -> dict[str, Any]:
    """Return a metadata-only receipt for a MobileGym source archive."""

    if not archive.is_file():
        raise ValueError(f"MobileGym source archive does not exist: {archive}")
    archive_identity = _identity(archive)
    prefix = f"mobilegym-{revision}/bench_env/splits/"
    payloads: dict[str, dict[str, Any]] = {}
    task_sets: dict[str, set[str]] = {}
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            members = {member.name: member for member in handle.getmembers()}
            for split in SPLITS:
                path = f"{prefix}{split}.txt"
                member = members.get(path)
                if member is None or not member.isfile():
                    raise ValueError(f"missing MobileGym split manifest: {path}")
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to read MobileGym split manifest: {path}")
                raw = extracted.read()
                payloads[split] = _split_payload(raw, name=split)
                task_sets[split] = {
                    line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()
                }
    except (OSError, tarfile.TarError, UnicodeDecodeError) as error:
        raise ValueError(f"unable to inspect MobileGym source archive: {archive}") from error

    overlap = sorted(task_sets["train"] & task_sets["test"])
    if overlap:
        raise ValueError(f"MobileGym train/test split overlap: {overlap[:3]}")
    train_test = task_sets["train"] | task_sets["test"]
    payload: dict[str, Any] = {
        "kind": "localagent_mobilegym_source_split_receipt",
        "schema_version": 1,
        "source": {
            "repository": REPOSITORY,
            "repository_url": REPOSITORY_URL,
            "revision": revision,
            "archive_url": ARCHIVE_URL,
            "archive": archive_identity,
            "license": {
                "code": "Apache-2.0",
                "benchmark_data": "CC-BY-NC-4.0",
                "policy": "split metadata only; no task text or benchmark content retained",
            },
        },
        "splits": payloads,
        "integrity": {
            "train_test_overlap": overlap,
            "train_test_unique_tasks": len(train_test),
            "official_test_tasks": len(task_sets["test"]),
            "official_train_tasks": len(task_sets["train"]),
            "all_split_task_ids_unique": len(set().union(*task_sets.values()))
            == sum(len(values) for values in task_sets.values()),
        },
        "localagent_adaptation": {
            "training_rows_added": 0,
            "simulator_runs": 0,
            "native_scores": 0,
            "task_text_retained": False,
            "claim_boundary": "Official split provenance only; no MobileGym simulator, judge, WebGPU model, or native score was run.",
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    receipt = profile(args.archive, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
