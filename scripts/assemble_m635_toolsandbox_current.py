#!/usr/bin/env python3
"""Compact the current m626 ToolSandbox base-matrix verifier run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CURRENT_CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assemble(raw_path: Path, output: Path) -> dict[str, Any]:
    raw = _load(raw_path)
    if raw.get("kind") != "localagent_toolsandbox_native_smoke":
        raise ValueError("unexpected ToolSandbox receipt kind")
    if raw.get("benchmark_id") != "toolsandbox":
        raise ValueError("unexpected ToolSandbox benchmark id")
    if raw.get("source_revision") != SOURCE_REVISION:
        raise ValueError("ToolSandbox source revision mismatch")
    if raw.get("checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("ToolSandbox report is not bound to current m626 checkpoint")
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 129:
        raise ValueError("expected the 129-scenario unaugmented ToolSandbox base matrix")
    category_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "exact": 0})
    scenario_summary: list[dict[str, Any]] = []
    for record in scenarios:
        if not isinstance(record, dict):
            raise ValueError("scenario record must be an object")
        exact = int(float(record.get("similarity", 0.0)) == 1.0)
        for category in record.get("categories", []):
            counts = category_totals[str(category)]
            counts["tasks"] += 1
            counts["exact"] += exact
        scenario_summary.append(
            {
                "scenario": record.get("scenario"),
                "categories": sorted(str(category) for category in record.get("categories", [])),
                "similarity": record.get("similarity"),
                "milestone_similarity": record.get("milestone_similarity"),
                "minefield_similarity": record.get("minefield_similarity"),
                "turn_count": record.get("turn_count"),
                "exception": record.get("exception"),
            }
        )
    payload: dict[str, Any] = {
        "kind": "localagent_m635_toolsandbox_current_native_base_receipt",
        "schema_version": 1,
        "benchmark_id": "toolsandbox",
        "environment_executed": raw.get("environment_executed"),
        "official_split_verified": raw.get("official_split_verified"),
        "task_count": raw.get("task_count"),
        "success_count": raw.get("success_count"),
        "success_rate": raw.get("success_rate"),
        "checkpoint": raw.get("checkpoint"),
        "source": {
            "url": raw.get("source_url"),
            "revision": raw.get("source_revision"),
            "runner": raw.get("runner"),
        },
        "protocol": {
            "name": raw.get("protocol"),
            "max_agent_turns": raw.get("max_agent_turns"),
            "user_simulator_executed": raw.get("user_simulator_executed"),
            "verifier_executed": raw.get("verifier_executed"),
            "external_api_called": raw.get("external_api_called"),
            "scenario_matrix": "ToolSandbox unaugmented base scenarios",
            "task_count": len(scenarios),
            "exception_count": sum(record.get("exception") is not None for record in scenarios),
        },
        "category_summary": {
            category: {
                **counts,
                "exact_rate": counts["exact"] / counts["tasks"] if counts["tasks"] else 0.0,
            }
            for category, counts in sorted(category_totals.items())
        },
        "scenarios": scenario_summary,
        "raw_report": _identity(raw_path),
        "claim_boundary": (
            "Native pinned ToolSandbox simulator and milestone-verifier evidence over the 129 "
            "unaugmented base scenarios. The upstream model-based user simulator, full 1,032 "
            "augmentation matrix, external APIs, and an official train/test split were not run; "
            "this is not an official ToolSandbox score or real-account email/Notion evidence."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.raw, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
