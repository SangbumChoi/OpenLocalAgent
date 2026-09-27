#!/usr/bin/env python
"""Profile a pinned AgentNet metadata JSONL snapshot without touching image/trajectory files."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.agentnet import AGENTNET_REVISION, AGENTNET_URL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            counts.update(str(item) for item in value)
        elif value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _action_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        frequencies = row.get("action_frequency")
        if not isinstance(frequencies, dict):
            continue
        for action, value in frequencies.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                counts[str(action)] += value
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=AGENTNET_REVISION)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"line {line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise SystemExit(f"line {line_number}: metadata row must be an object")
            for key in ("task_id", "system", "domains", "step_num"):
                if key not in row:
                    raise SystemExit(f"line {line_number}: missing required metadata key {key!r}")
            rows.append(row)
    task_ids = [str(row["task_id"]) for row in rows]
    payload: dict[str, Any] = {
        "kind": "localagent_agentnet_metadata_profile",
        "schema_version": 1,
        "dataset": "AgentNet",
        "dataset_url": AGENTNET_URL,
        "source_revision": args.revision,
        "source": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": _sha256(args.input),
            "format": "metadata_jsonl",
            "trajectory_or_image_payload_consumed": False,
        },
        "rows": len(rows),
        "unique_task_ids": len(set(task_ids)),
        "duplicate_task_id_rows": len(task_ids) - len(set(task_ids)),
        "systems": _counter(rows, "system"),
        "domains": _counter(rows, "domains"),
        "applications": _counter(rows, "applications"),
        "websites": _counter(rows, "websites"),
        "releases": _counter(rows, "release"),
        "action_frequency": _action_counter(rows),
        "step_num": {
            "min": min((int(row["step_num"]) for row in rows), default=0),
            "max": max((int(row["step_num"]) for row in rows), default=0),
        },
        "claim_boundary": "Metadata inventory only; no AgentNetBench score and no trajectory/image training data.",
    }
    payload["profile_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
