#!/usr/bin/env python
"""Profile the public AndroidWorld task metadata without starting an emulator.

The upstream task metadata is a small, public inventory of task templates.  This command binds it
to an AndroidWorld commit, summarizes difficulty/tags/step budgets, and writes a receipt.  It never
imports AndroidWorld, installs APKs, invokes ``adb``, reads screenshots, or treats task metadata as
training data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


DATASET = "google-research/android_world"
SOURCE_URL = "https://github.com/google-research/android_world"
REVISION = "3e50888527ef9f29b9157ecd537e408008bb1c85"
METADATA_PATH = "android_world/task_metadata.json"
LICENSE = "Apache-2.0"

_PREFIXES = (
    "TurnOffWifiAndTurnOnBluetooth",
    "TurnOnWifiAndOpenApp",
    "SimpleCalendar",
    "SimpleDrawPro",
    "SimpleSms",
    "SportsTracker",
    "AudioRecorder",
    "Browser",
    "Camera",
    "Clock",
    "Contacts",
    "Expense",
    "Files",
    "Markor",
    "Notes",
    "OpenApp",
    "OsmAnd",
    "Recipe",
    "Retro",
    "SaveCopy",
    "System",
    "Tasks",
    "Vlc",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category(task_name: str) -> str:
    for prefix in _PREFIXES:
        if task_name.startswith(prefix):
            return prefix
    return "other"


def _counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(values).items(), key=lambda pair: (-pair[1], pair[0])))


def profile(path: Path, *, revision: str = REVISION) -> dict[str, Any]:
    """Return a deterministic inventory receipt for ``task_metadata.json``."""

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read AndroidWorld metadata: {path}") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("AndroidWorld metadata must be a non-empty JSON list")

    tasks: list[dict[str, Any]] = []
    steps: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"metadata row {index} must be an object")
        required = {"task_name", "task_template", "difficulty", "tags", "optimal_steps"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"metadata row {index} is missing {missing}")
        task_name = row["task_name"]
        template = row["task_template"]
        difficulty = row["difficulty"]
        tags = row["tags"]
        if (
            not isinstance(task_name, str)
            or not task_name
            or not isinstance(template, str)
            or not template
            or not isinstance(difficulty, str)
            or not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
        ):
            raise ValueError(f"metadata row {index} has invalid field types")
        try:
            optimal_steps = int(row["optimal_steps"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"metadata row {index} optimal_steps is not an integer") from error
        if optimal_steps < 1:
            raise ValueError(f"metadata row {index} optimal_steps must be positive")
        steps.append(optimal_steps)
        tasks.append(
            {
                "task_name": task_name,
                "category": _category(task_name),
                "difficulty": difficulty,
                "tags": sorted(set(tags)),
                "optimal_steps": optimal_steps,
                "template_has_parameters": "{" in template and "}" in template,
            }
        )

    names = [task["task_name"] for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError("AndroidWorld task_name values must be unique")
    tags = [tag for task in tasks for tag in task["tags"] if tag]
    difficulties = [task["difficulty"] for task in tasks]
    payload: dict[str, Any] = {
        "kind": "localagent_androidworld_metadata_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_file_url": f"{SOURCE_URL}/blob/{revision}/{METADATA_PATH}",
        "revision": revision,
        "license": LICENSE,
        "source": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "format": "androidworld_task_metadata_json",
            "metadata_only": True,
            "emulator_executed": False,
            "adb_invoked": False,
        },
        "coverage": {
            "task_templates": len(tasks),
            "unique_task_names": len(set(names)),
            "difficulties": _counter(difficulties),
            "categories": _counter([task["category"] for task in tasks]),
            "tags": _counter(tags),
            "parameterized_tasks": sum(task["template_has_parameters"] for task in tasks),
            "optimal_steps": {
                "min": min(steps),
                "max": max(steps),
                "mean": statistics.mean(steps),
                "median": statistics.median(steps),
            },
        },
        "task_inventory": tasks,
        "runtime_requirements": {
            "android_emulator": True,
            "adb": True,
            "apk_setup": True,
            "task_verifiers": "upstream AndroidWorld environment reward/checkpointer",
        },
        "claim_boundary": (
            "Public task-metadata inventory only; no emulator, APK, adb, screenshot, task "
            "verifier, AndroidWorld reward, leaderboard score, or training artifact is claimed. "
            "All task templates remain evaluation-only."
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
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite metadata receipt")
    payload = profile(args.input, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
