"""Fail-closed aggregation for normalized BrowserGym/Gymnasium episode logs.

BrowserGym itself is an interactive environment, not a single static result-file format. A local
runner can serialize each Gymnasium step into the small JSONL contract consumed here:
`task_id`, `seed`, and a non-empty `steps` list containing `action`, `reward`,
`terminated`, and `truncated`. This module validates that contract and computes task/reward
metrics without importing Playwright or launching a browser.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _expected_cases(expected_cases: Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_cases is None:
        return None
    if isinstance(expected_cases, (str, bytes)) or not expected_cases:
        raise ValueError("expected_cases must be a non-empty sequence")
    cases = tuple(sorted(_text(case, label="expected case") for case in expected_cases))
    if len(set(cases)) != len(cases):
        raise ValueError("expected_cases contains duplicates")
    return cases


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read BrowserGym episode JSONL: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"BrowserGym episode JSONL is empty: {path}")
    return rows


def _case_key(task_id: str, seed: int) -> str:
    return f"{task_id}@{seed}"


def _validate_episode(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    task_id = _text(row.get("task_id"), label=f"{label}.task_id")
    seed = row.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{label}.seed must be an integer")
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{label}.steps must be a non-empty list")
    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        step_label = f"{label}.steps[{index}]"
        if not isinstance(step, Mapping):
            raise ValueError(f"{step_label} must be an object")
        action = _text(step.get("action"), label=f"{step_label}.action")
        reward = _finite(step.get("reward"), label=f"{step_label}.reward")
        terminated = _bool(step.get("terminated"), label=f"{step_label}.terminated")
        truncated = _bool(step.get("truncated"), label=f"{step_label}.truncated")
        error = step.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"{step_label}.error must be text or null")
        normalized_steps.append(
            {
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "error": error,
            }
        )
    final_reward = _finite(
        row.get("final_reward", normalized_steps[-1]["reward"]),
        label=f"{label}.final_reward",
    )
    if abs(final_reward - normalized_steps[-1]["reward"]) > 1e-9:
        raise ValueError(f"{label}.final_reward must equal the last step reward")
    terminated = _bool(
        row.get("terminated", normalized_steps[-1]["terminated"]),
        label=f"{label}.terminated",
    )
    truncated = _bool(
        row.get("truncated", normalized_steps[-1]["truncated"]),
        label=f"{label}.truncated",
    )
    if not (terminated or truncated):
        raise ValueError(f"{label} must end with terminated or truncated true")
    if terminated != normalized_steps[-1]["terminated"]:
        raise ValueError(f"{label}.terminated must match the last step")
    if truncated != normalized_steps[-1]["truncated"]:
        raise ValueError(f"{label}.truncated must match the last step")
    return {
        "task_id": task_id,
        "seed": seed,
        "case": _case_key(task_id, seed),
        "steps": normalized_steps,
        "final_reward": final_reward,
        "terminated": terminated,
        "truncated": truncated,
        "action_errors": sum(step["error"] is not None for step in normalized_steps),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def aggregate_browsergym_episodes(
    episodes_jsonl: str | Path,
    *,
    expected_cases: Sequence[str] | None = None,
    source_revision: str | None = None,
    miniwob_revision: str | None = None,
    runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Aggregate normalized BrowserGym episode logs and verify exact case coverage."""

    path = Path(episodes_jsonl)
    expected = _expected_cases(expected_cases)
    rows = [
        _validate_episode(row, label=f"episodes[{index}]")
        for index, row in enumerate(_jsonl(path))
    ]
    cases = [row["case"] for row in rows]
    if len(set(cases)) != len(cases):
        raise ValueError("BrowserGym episode log contains duplicate task/seed cases")
    discovered = tuple(sorted(cases))
    expected_set = set(expected or ())
    discovered_set = set(discovered)
    missing = sorted(expected_set - discovered_set) if expected is not None else []
    unexpected = sorted(discovered_set - expected_set) if expected is not None else []
    complete = bool(expected is not None and not missing and not unexpected)
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    task_summary = {
        task: {
            "cases": len(task_rows),
            "mean_final_reward": _mean(task_rows, "final_reward"),
            "success_rate": sum(row["final_reward"] >= 1.0 for row in task_rows)
            / len(task_rows),
            "mean_steps": sum(len(row["steps"]) for row in task_rows) / len(task_rows),
            "action_errors": sum(row["action_errors"] for row in task_rows),
        }
        for task, task_rows in sorted(by_task.items())
    }
    source_bytes = path.read_bytes()
    return {
        "kind": "localagent_browsergym_episode_aggregate",
        "schema_version": 1,
        "episodes_jsonl": str(path),
        "episodes_bytes": len(source_bytes),
        "episodes_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_revision": source_revision,
        "miniwob_revision": miniwob_revision,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "cases": len(rows),
        "tasks": len(task_summary),
        "overall": {
            "mean_final_reward": _mean(rows, "final_reward"),
            "success_rate": sum(row["final_reward"] >= 1.0 for row in rows) / len(rows),
            "mean_steps": sum(len(row["steps"]) for row in rows) / len(rows),
            "action_errors": sum(row["action_errors"] for row in rows),
            "terminated_cases": sum(row["terminated"] for row in rows),
            "truncated_cases": sum(row["truncated"] for row in rows),
        },
        "by_task": task_summary,
        "completeness": {
            "verified": complete,
            "expected_cases": list(expected) if expected is not None else None,
            "discovered_cases": list(discovered),
            "missing_cases": missing,
            "unexpected_cases": unexpected,
        },
        "status": "complete" if complete else "incomplete",
        "claim_scope": (
            "Local aggregation of normalized BrowserGym/Gymnasium episode logs; exact case "
            "coverage is required for status=complete. This is not an official BrowserGym, "
            "WebArena, WorkArena, or MiniWoB leaderboard score and does not launch a browser."
        ),
    }


__all__ = ["aggregate_browsergym_episodes"]
