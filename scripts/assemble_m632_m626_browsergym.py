#!/usr/bin/env python3
"""Compact the full current-checkpoint BrowserGym/MiniWoB run into a paper receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CURRENT_CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


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
    if raw.get("kind") != "localagent_browsergym_native_eval":
        raise ValueError("unexpected BrowserGym receipt kind")
    if raw.get("benchmark_id") != "browsergym_miniwob":
        raise ValueError("unexpected BrowserGym benchmark id")
    if raw.get("checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("BrowserGym report is not bound to current m626 checkpoint")
    if raw.get("environment_executed") is not True or raw.get("official_split_verified") is not True:
        raise ValueError("BrowserGym report is not a complete official execution")
    cases = raw.get("cases")
    if not isinstance(cases, list) or len(cases) != 240:
        raise ValueError("BrowserGym report must contain all 240 cases")
    family_total: Counter[str] = Counter()
    family_success: Counter[str] = Counter()
    family_steps: defaultdict[str, int] = defaultdict(int)
    grounded_steps = 0
    noop_steps = 0
    action_errors = 0
    for case in cases:
        task = str(case.get("task", ""))
        family_total[task] += 1
        if case.get("success") is True:
            family_success[task] += 1
        steps = case.get("steps", [])
        family_steps[task] += len(steps) if isinstance(steps, list) else 0
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("grounded") is True:
                    grounded_steps += 1
                else:
                    noop_steps += 1
                if step.get("info_action_error"):
                    action_errors += 1
    result = {
        "task_count": len(cases),
        "passed_tasks": sum(family_success.values()),
        "failed_tasks": len(cases) - sum(family_success.values()),
        "success_rate": raw.get("success_rate"),
        "grounded_steps": grounded_steps,
        "noop_or_ungrounded_steps": noop_steps,
        "action_errors": action_errors,
        "successful_task_families": dict(sorted(family_success.items())),
        "task_family_summary": {
            task: {
                "episodes": family_total[task],
                "passed": family_success[task],
                "success_rate": family_success[task] / family_total[task],
                "model_steps": family_steps[task],
            }
            for task in sorted(family_total)
        },
        "observation_mode": "accessibility_tree_text",
        "vision_used": False,
        "coordinate_fallback": raw.get("coordinate_fallback"),
        "semantic_fallback": raw.get("semantic_fallback"),
    }
    payload: dict[str, Any] = {
        "kind": "localagent_m632_browsergym_m626_native_full_receipt",
        "schema_version": 1,
        "benchmark_id": raw["benchmark_id"],
        "environment_executed": raw["environment_executed"],
        "official_split_verified": raw["official_split_verified"],
        "task_count": raw["task_count"],
        "success_rate": raw["success_rate"],
        "checkpoint": raw["checkpoint"],
        "browsergym": raw.get("browsergym"),
        "runtime": raw.get("runtime"),
        "task_plan": raw.get("task_plan"),
        "result": result,
        "raw_report": _identity(raw_path),
        "claim_boundary": raw.get("claim_boundary"),
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
