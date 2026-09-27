"""Dependency-free scoring for normalized AndroidControl/AITW action trajectories.

The scorer consumes the text-first ``localagent_v1`` rows emitted by the mobile adapters and a
prediction stream from a model or WebGPU harness.  It deliberately does not infer environment
state, inspect screenshots, or claim an AndroidControl/AITW/AndroidWorld leaderboard result.  A
later emulator runner can use this action contract as its per-step diagnostic while reporting
official environment rewards separately.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_COORDINATE_ACTIONS = frozenset({"mobile_click", "mobile_long_press"})


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return unicodedata.normalize("NFKC", value)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _arguments(value: object, *, label: str) -> dict[str, Any]:
    mapping = _mapping(value, label=label)
    return {str(key): item for key, item in mapping.items()}


def _call(value: object, *, label: str) -> tuple[str, dict[str, Any]]:
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str):
        return _text(value[0], label=f"{label}.name"), _arguments(value[1], label=f"{label}.arguments")
    mapping = _mapping(value, label=label)
    if isinstance(mapping.get("name"), str):
        name = mapping["name"]
        arguments = mapping.get("arguments", {})
    elif isinstance(mapping.get("tool"), str):
        name = mapping["tool"]
        arguments = mapping.get("args", mapping.get("arguments", {}))
    else:
        raise ValueError(f"{label} must contain a tool/name string")
    return _text(name, label=f"{label}.name"), _arguments(arguments, label=f"{label}.arguments")


def _expected_calls(row: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    messages = row.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("mobile row messages must be a sequence")
    calls: list[tuple[str, dict[str, Any]]] = []
    for index, raw_message in enumerate(messages):
        message = _mapping(raw_message, label=f"messages[{index}]")
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise ValueError(f"messages[{index}].tool_calls must be a sequence")
        for call_index, raw_call in enumerate(raw_calls):
            calls.append(_call(raw_call, label=f"messages[{index}].tool_calls[{call_index}]"))
    return calls


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("arguments contain a non-finite number")
    return value


def _exact_arguments(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    # ``target_bbox`` is optional evaluator metadata, not an argument emitted to the device.
    expected_action_args = {key: value for key, value in expected.items() if key != "target_bbox"}
    return _canonical(expected_action_args) == _canonical(predicted)


def _coordinate_score(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> float | None:
    if not {"x", "y"}.issubset(expected) or not {"x", "y"}.issubset(predicted):
        return None
    expected_x = _finite_number(expected["x"], label="expected.x")
    expected_y = _finite_number(expected["y"], label="expected.y")
    predicted_x = _finite_number(predicted["x"], label="predicted.x")
    predicted_y = _finite_number(predicted["y"], label="predicted.y")
    distance = math.hypot(predicted_x - expected_x, predicted_y - expected_y)
    scale = max(abs(expected_x), abs(expected_y), 1.0)
    return max(0.0, 1.0 - distance / scale)


def _grounded_score(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> float | None:
    """Score an optional evaluator-provided target box without reading pixels."""

    target = expected.get("target_bbox")
    if target is None:
        return None
    if not isinstance(target, Sequence) or isinstance(target, (str, bytes)) or len(target) != 4:
        raise ValueError("expected.target_bbox must be [left, top, right, bottom]")
    if not {"x", "y"}.issubset(predicted):
        return 0.0
    left, top, right, bottom = (
        _finite_number(value, label="expected.target_bbox[]") for value in target
    )
    x = _finite_number(predicted["x"], label="predicted.x")
    y = _finite_number(predicted["y"], label="predicted.y")
    return float(left <= x <= right and top <= y <= bottom)


def score_mobile_actions(
    expected_actions: Sequence[object],
    predicted_actions: Sequence[object],
) -> dict[str, Any]:
    """Score a normalized mobile action stream with fail-closed structural validation."""

    if isinstance(expected_actions, (str, bytes)) or isinstance(predicted_actions, (str, bytes)):
        raise ValueError("action streams must be sequences, not text")
    expected = [_call(value, label=f"expected_actions[{index}]") for index, value in enumerate(expected_actions)]
    predicted = [_call(value, label=f"predicted_actions[{index}]") for index, value in enumerate(predicted_actions)]
    if not expected:
        raise ValueError("expected action stream must not be empty")

    compared = min(len(expected), len(predicted))
    type_matches = 0
    exact_matches = 0
    coordinate_scores: list[float] = []
    grounded_scores: list[float] = []
    per_step: list[dict[str, Any]] = []
    for index in range(compared):
        expected_name, expected_args = expected[index]
        predicted_name, predicted_args = predicted[index]
        type_match = expected_name == predicted_name
        exact_match = type_match and _exact_arguments(expected_args, predicted_args)
        if type_match:
            type_matches += 1
        if exact_match:
            exact_matches += 1
        coordinate = (
            _coordinate_score(expected_args, predicted_args)
            if type_match and expected_name in _COORDINATE_ACTIONS
            else None
        )
        grounded = (
            _grounded_score(expected_args, predicted_args)
            if type_match and expected_name in _COORDINATE_ACTIONS
            else None
        )
        if coordinate is not None:
            coordinate_scores.append(coordinate)
        if grounded is not None:
            grounded_scores.append(grounded)
        per_step.append(
            {
                "index": index,
                "expected_tool": expected_name,
                "predicted_tool": predicted_name,
                "tool_match": type_match,
                "action_exact": exact_match,
                "coordinate_score": coordinate,
                "grounded_score": grounded,
            }
        )

    denominator = len(expected)
    return {
        "expected_actions": len(expected),
        "predicted_actions": len(predicted),
        "compared_actions": compared,
        "tool_accuracy": type_matches / denominator,
        "action_exact_accuracy": exact_matches / denominator,
        "coordinate_score_mean": sum(coordinate_scores) / len(coordinate_scores)
        if coordinate_scores
        else None,
        "grounded_score_mean": sum(grounded_scores) / len(grounded_scores)
        if grounded_scores
        else None,
        "trajectory_exact": len(expected) == len(predicted) and exact_matches == denominator,
        "per_step": per_step,
        "claim_scope": (
            "offline normalized AndroidControl/AITW action diagnostic; not an official device, "
            "AndroidWorld, AndroidControl, or AITW benchmark score"
        ),
    }


def score_mobile_row(row: Mapping[str, Any], predicted_actions: Sequence[object]) -> dict[str, Any]:
    """Extract expected tool calls from one normalized row and score predictions."""

    expected = _expected_calls(_mapping(row, label="row"))
    result = score_mobile_actions(expected, predicted_actions)
    if "record_id" in row:
        result["record_id"] = row["record_id"]
    quality = row.get("quality")
    if isinstance(quality, Mapping):
        result["source_family"] = quality.get("source_family")
        result["source_split"] = quality.get("source_split")
    return result


def serialize_score(result: Mapping[str, Any]) -> str:
    """Serialize a score with stable key ordering for receipts and tests."""

    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["score_mobile_actions", "score_mobile_row", "serialize_score"]
