#!/usr/bin/env python3
"""Audit the public ClawsBench productivity-task metadata without executing services."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = "benchflow/ClawsBench"
DATASET_URL = "https://huggingface.co/datasets/benchflow/ClawsBench"
SOURCE_URL = "https://github.com/benchflow-ai/ClawsBench"
REVISION = "e7c45cc9ff486502176267c1294ac5809cf0700a"
LICENSE = "CC-BY-NC-SA-4.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def audit(tasks_path: Path, experiments_path: Path, results_path: Path) -> dict[str, Any]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, dict) or len(tasks) != 44:
        raise ValueError("expected the public 44-task ClawsBench metadata release")
    categories = Counter(str(row.get("category", "unknown")) for row in tasks.values())
    services = Counter(
        service
        for row in tasks.values()
        for service in row.get("services", [])
    )
    safety = sum(bool(row.get("is_safety")) for row in tasks.values())
    experiment_summary = {
        name: {
            "tasks": value.get("n_tasks"),
            "repeats": value.get("repeats"),
            "environment": value.get("environment"),
            "git_hash": value.get("git_hash"),
        }
        for name, value in sorted(experiments.items())
    }
    body: dict[str, Any] = {
        "kind": "localagent_clawsbench_source_audit",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "source_url": SOURCE_URL,
            "revision": REVISION,
            "license": LICENSE,
            "files": {
                "tasks": _identity(tasks_path),
                "experiments": _identity(experiments_path),
                "results": _identity(results_path),
            },
        },
        "task_metadata": {
            "tasks": len(tasks),
            "safety_tasks": safety,
            "non_safety_tasks": len(tasks) - safety,
            "categories": dict(sorted(categories.items())),
            "services": dict(sorted(services.items())),
            "single_service_tasks": sum(row.get("n_services") == 1 for row in tasks.values()),
            "multi_service_tasks": sum(row.get("n_services") and row.get("n_services") > 1 for row in tasks.values()),
        },
        "experiments": experiment_summary,
        "evaluation_boundary": {
            "public_task_metadata": True,
            "public_trajectory_traces": True,
            "public_environment_and_verifiers": False,
            "official_static_train_test_split": False,
            "train_policy": "eval_only",
            "reason": (
                "The public release exposes task metadata and traces/results, while the "
                "Dockerized mock services, verifier state, and reproducible task runner are not "
                "included in this snapshot. The task metadata therefore cannot support a LocalAgent "
                "teacher-forced gold-action target or native score."
            ),
        },
        "claim_boundary": (
            "Source/protocol audit only. No Gmail, Calendar, Docs, Drive, Slack service, credentials, "
            "Docker environment, verifier, or external side effect was executed; no ClawsBench score "
            "is claimed."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = audit(args.tasks, args.experiments, args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
