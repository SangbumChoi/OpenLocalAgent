#!/usr/bin/env python3
"""Seal a current-checkpoint MCPMark filesystem easy-suite replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assemble(*, source: Path, checkpoint: Path) -> dict[str, Any]:
    raw = _load(source)
    if raw.get("kind") != "localagent_mcpmark_native_filesystem_easy_current_checkpoint":
        raise ValueError("source receipt kind mismatch")
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("revision") != SOURCE_REVISION:
        raise ValueError("MCPMark source revision mismatch")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("source model identity missing")
    expected = _identity(checkpoint)
    if model.get("sha256") != expected["sha256"]:
        raise ValueError("source receipt is not bound to the supplied checkpoint")
    summary = raw.get("summary")
    results = raw.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list) or not results:
        raise ValueError("source results/summary missing")
    task_count = int(summary.get("tasks", 0))
    passes = int(summary.get("verifier_passes", -1))
    if task_count != len(results) or task_count < 1 or passes < 0 or passes > task_count:
        raise ValueError("invalid task summary")
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("verifier_exit_code"), int):
            raise ValueError("each task must carry an integer verifier exit code")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_current_filesystem_easy_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "environment_executed": True,
        "official_split_verified": False,
        "task_count": task_count,
        "success_rate": passes / task_count,
        "checkpoint_sha256": expected["sha256"],
        "source": {
            "dataset": dataset,
            "receipt": _identity(source),
            "official_split_verified": False,
        },
        "model": expected,
        "environment": raw.get("environment"),
        "task_source": raw.get("task_source"),
        "results": results,
        "summary": {
            **summary,
            "task_count": task_count,
            "success_rate": passes / task_count,
        },
        "decision": {
            "native_task_passes": passes,
            "native_task_count": task_count,
            "official_split_verified": False,
            "promotion": "blocked_pending_official_mcpmark_split_and_broader_tasks",
            "claim_boundary": (
                "The exact current LocalAgent checkpoint ran ten isolated public MCPMark easy "
                "filesystem fixtures through a real stdio server and pinned verifiers; two "
                "passed. This is not an official MCPMark split or leaderboard score, and it "
                "does not establish email, Notion, browser, desktop, user-simulator, or "
                "multi-service completion."
            ),
        },
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")
    payload = assemble(source=args.source, checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "receipt_self_sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
