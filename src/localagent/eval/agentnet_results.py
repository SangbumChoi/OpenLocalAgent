"""Fail-closed aggregation for AgentNet/OpenCUA computer-use action predictions.

AgentNetBench is an offline action evaluator: it scores low-level desktop actions against
ground-truth trajectories and does not itself launch a desktop. This module joins one public
ground-truth JSONL file with one prediction JSONL file, delegates per-record scoring to the
dependency-free AgentNet-compatible scorer, and emits a reproducible receipt. It never opens
screenshots, starts an OS runner, or claims an official AgentNetBench leaderboard score.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from localagent.data.agentnet import parse_pyautogui_actions
from localagent.eval.agentnet import score_agentnet_record

_SUCCESS_EPSILON = 1e-6


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _expected_ids(expected_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_ids is None:
        return None
    if isinstance(expected_ids, (str, bytes)) or not expected_ids:
        raise ValueError("expected_ids must be a non-empty sequence")
    values = tuple(sorted(_text(value, label="expected task id") for value in expected_ids))
    if len(set(values)) != len(values):
        raise ValueError("expected_ids contains duplicates")
    return values


def _jsonl(path: Path, *, label: str) -> tuple[list[Mapping[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read {label}: {path}") from error
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
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
        raise ValueError(f"{label} is empty: {path}")
    return rows, payload


def _platform(row: Mapping[str, Any]) -> str:
    for key in ("platform", "os", "device"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("platform", "os", "device"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().casefold()
    return "unknown"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _prediction_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[list[object], str]]:
    predictions: dict[str, tuple[list[object], str]] = {}
    for index, row in enumerate(rows):
        task_id = _text(row.get("task_id"), label=f"predictions[{index}].task_id")
        raw_actions = row.get("predicted_actions", row.get("actions"))
        if not isinstance(raw_actions, list):
            raise ValueError(
                f"predictions[{index}].predicted_actions must be a list"
            )
        if task_id in predictions:
            raise ValueError(f"duplicate prediction task id: {task_id}")
        predictions[task_id] = (list(raw_actions), _platform(row))
    return predictions


def _ground_truth_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], str]]:
    records: dict[str, tuple[Mapping[str, Any], str]] = {}
    for index, row in enumerate(rows):
        task_id = _text(row.get("task_id"), label=f"ground_truth[{index}].task_id")
        if task_id in records:
            raise ValueError(f"duplicate ground-truth task id: {task_id}")
        records[task_id] = (row, _platform(row))
    return records


def _code_action_to_ground_truth(
    action: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Project current AgentNet pyautogui-code rows into AgentNetBench action objects."""

    action_type = _text(action.get("action_type"), label=f"{label}.action_type")
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{label}.arguments must be an object")
    if action_type in {
        "click",
        "doubleClick",
        "rightClick",
        "middleClick",
        "tripleClick",
        "moveTo",
        "dragTo",
    }:
        params: dict[str, Any] = {
            "position": {"x": arguments.get("x"), "y": arguments.get("y")}
        }
    elif action_type == "write":
        params = {"text": arguments.get("text", arguments.get("message", ""))}
    elif action_type in {"press", "hotkey"}:
        params = {"keys": arguments.get("keys", [])}
    elif action_type in {"scroll", "hscroll"}:
        params = {"amount": arguments.get("clicks", arguments.get("amount", 0))}
    elif action_type == "wait":
        params = {"seconds": arguments.get("seconds", 0)}
    elif action_type == "terminate":
        params = {"status": arguments.get("status", "success")}
    else:
        raise ValueError(f"{label} uses unsupported action {action_type!r}")
    return {"type": action_type, "params": params, "metadata": {}}


def _scoring_record(record: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    """Accept both AgentNetBench action objects and current public pyautogui-code exports."""

    if isinstance(record.get("ground_truth_actions"), list):
        return record
    raw_steps = record.get("steps")
    if raw_steps is None:
        raw_steps = record.get("traj")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{label} must contain steps or traj")
    projected_actions: list[dict[str, Any]] = []
    for step_index, step in enumerate(raw_steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"{label}.steps[{step_index}] must be an object")
        explicit = step.get("ground_truth_actions")
        if isinstance(explicit, list):
            projected_actions.extend(explicit)
            continue
        value = step.get("value") if isinstance(step.get("value"), Mapping) else step
        source = value.get("code") if isinstance(value, Mapping) else None
        if not isinstance(source, str) or not source.strip():
            source = value.get("action") if isinstance(value, Mapping) else None
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"{label}.steps[{step_index}] has no ground_truth_actions or action code"
            )
        parsed = parse_pyautogui_actions(source, label=f"{label}.steps[{step_index}].action")
        projected_actions.extend(
            _code_action_to_ground_truth(
                action,
                label=f"{label}.steps[{step_index}].actions[{action_index}]",
            )
            for action_index, action in enumerate(parsed)
        )
    if not projected_actions:
        raise ValueError(f"{label} contains no ground-truth actions")
    return {"steps": [{"ground_truth_actions": projected_actions}]}


def _score_summary(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = [float(row["total"]) for row in scored]
    action_values: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        for action_type, value in row["actions"].items():
            action_values[action_type].append(float(value))
    return {
        "tasks": len(scored),
        "mean_total": _mean(totals),
        "success_rate": sum(value >= 1.0 - _SUCCESS_EPSILON for value in totals)
        / len(scored)
        if scored
        else 0.0,
        "exact_trajectory_rate": sum(value == 1.0 for value in totals) / len(scored)
        if scored
        else 0.0,
        "mean_ground_truth_actions": _mean(
            [float(row["ground_truth_count"]) for row in scored]
        ),
        "mean_predicted_actions": _mean(
            [float(row["predicted_count"]) for row in scored]
        ),
        "first_action_type_rate": _mean(
            [float(row["first_action_type_match"]) for row in scored]
        ),
        "mean_action_count_penalty": _mean(
            [float(row.get("action_count_penalty", 1.0)) for row in scored]
        ),
        "by_action": {
            action_type: {
                "tasks_with_action": len(values),
                "mean_score": _mean(values),
            }
            for action_type, values in sorted(action_values.items())
        },
    }


def aggregate_agentnet_results(
    ground_truth_jsonl: str | Path,
    predictions_jsonl: str | Path,
    *,
    expected_ids: Sequence[str] | None = None,
    source_revision: str | None = None,
    runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Aggregate AgentNet ground truth and predictions with exact task coverage."""

    ground_truth_path = Path(ground_truth_jsonl)
    predictions_path = Path(predictions_jsonl)
    ground_rows, ground_bytes = _jsonl(ground_truth_path, label="AgentNet ground truth JSONL")
    prediction_rows, prediction_bytes = _jsonl(
        predictions_path, label="AgentNet prediction JSONL"
    )
    expected = _expected_ids(expected_ids)
    ground_truth = _ground_truth_map(ground_rows)
    predictions = _prediction_map(prediction_rows)
    ground_ids = set(ground_truth)
    prediction_ids = set(predictions)
    missing_predictions = sorted(ground_ids - prediction_ids)
    unexpected_predictions = sorted(prediction_ids - ground_ids)
    expected_set = set(expected or ())
    missing_expected = sorted(expected_set - ground_ids)
    unexpected_ground = sorted(ground_ids - expected_set) if expected is not None else []
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "AgentNet prediction coverage does not match ground truth: "
            f"missing={missing_predictions}, unexpected={unexpected_predictions}"
        )
    scored: list[dict[str, Any]] = []
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id in sorted(ground_truth):
        record, ground_platform = ground_truth[task_id]
        actions, prediction_platform = predictions[task_id]
        score = score_agentnet_record(
            _scoring_record(record, label=f"ground_truth[{task_id}]"),
            actions,
        )
        normalized = {
            "task_id": task_id,
            "platform": (
                prediction_platform
                if prediction_platform != "unknown"
                else ground_platform
            ),
            **score,
        }
        scored.append(normalized)
        by_platform[normalized["platform"]].append(normalized)
    complete = bool(
        expected is not None
        and not missing_expected
        and not unexpected_ground
        and not missing_predictions
        and not unexpected_predictions
    )
    source_hash = hashlib.sha256()
    source_hash.update(ground_bytes)
    source_hash.update(prediction_bytes)
    return {
        "kind": "localagent_agentnet_result_aggregate",
        "schema_version": 1,
        "ground_truth": {
            "path": str(ground_truth_path),
            "bytes": len(ground_bytes),
            "sha256": hashlib.sha256(ground_bytes).hexdigest(),
        },
        "predictions": {
            "path": str(predictions_path),
            "bytes": len(prediction_bytes),
            "sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        },
        "source_sha256": source_hash.hexdigest(),
        "source_revision": source_revision,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "overall": _score_summary(scored),
        "by_platform": {
            platform: _score_summary(rows)
            for platform, rows in sorted(by_platform.items())
        },
        "records": scored,
        "completeness": {
            "verified": complete,
            "expected_ids": list(expected) if expected is not None else None,
            "ground_truth_ids": sorted(ground_ids),
            "prediction_ids": sorted(prediction_ids),
            "missing_expected_ids": missing_expected,
            "unexpected_ground_truth_ids": unexpected_ground,
            "missing_predictions": missing_predictions,
            "unexpected_predictions": unexpected_predictions,
        },
        "status": "complete" if complete else "incomplete",
        "claim_scope": (
            "Local aggregation using the dependency-free AgentNetBench-compatible action scorer. "
            "This receipt is offline and does not open screenshots, launch Windows/macOS/Ubuntu, "
            "or constitute an official AgentNetBench leaderboard result."
        ),
    }


__all__ = ["aggregate_agentnet_results"]
