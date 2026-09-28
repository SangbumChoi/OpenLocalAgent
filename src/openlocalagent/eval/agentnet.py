"""Dependency-free offline scoring compatible with the public AgentNetBench action protocol.

This scorer intentionally evaluates only action records already present on disk.  It never opens a
desktop, reads screenshots, or claims an official AgentNetBench leaderboard result.  The formulas
mirror the public evaluator's coordinate, text, keyboard, scroll, action-count, and termination
rules so a future visual/VM bridge can compare a WebGPU prediction stream before invoking the
upstream harness.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_COORD_THRESHOLD = 0.01 * 2**0.5
_ALPHA = 120.0
_WRITE_SIMILARITY_THRESHOLD = 0.8

_TOOL_TO_ACTION = {
    "agentnet_click": "click",
    "agentnet_double_click": "doubleClick",
    "agentnet_right_click": "rightClick",
    "agentnet_middle_click": "middleClick",
    "agentnet_triple_click": "tripleClick",
    "agentnet_move_cursor": "moveTo",
    "agentnet_drag": "dragTo",
    "agentnet_scroll": "scroll",
    "agentnet_hscroll": "hscroll",
    "agentnet_type_text": "write",
    "agentnet_key_press": "press",
    "agentnet_hotkey": "hotkey",
    "agentnet_wait": "wait",
    "agentnet_terminate": "terminate",
}


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _normalize_keys(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).casefold() for item in value]
    if value in (None, ""):
        return []
    return [str(value).casefold()]


def _direction(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1 if value > 0 else -1 if value < 0 else 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"up", "positive", "right"}:
            return 1
        if normalized in {"down", "negative", "left"}:
            return -1
        try:
            return _direction(float(normalized))
        except ValueError:
            return 0
    return 0


def _finite_pair(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-value coordinate")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair):
        raise ValueError(f"{label} must contain finite coordinates")
    return pair


def _action_position(action: Mapping[str, Any], *, label: str) -> tuple[float, float]:
    params = action.get("params")
    if not isinstance(params, Mapping):
        raise ValueError(f"{label}.params must be an object")
    position = params.get("position")
    if not isinstance(position, Mapping):
        raise ValueError(f"{label}.params.position must be an object")
    return _finite_pair((position.get("x"), position.get("y")), label=f"{label}.position")


def _point_in_bboxes(point: Sequence[float], ground_truth: Mapping[str, Any]) -> bool:
    """Return whether a normalized point is inside an evaluator-provided target box."""

    metadata = ground_truth.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    bboxes = metadata.get("bboxes", [])
    if not isinstance(bboxes, Sequence) or isinstance(bboxes, (str, bytes)):
        return False
    x, y = point
    for bbox in bboxes:
        if not isinstance(bbox, Mapping):
            continue
        rel_bbox = bbox.get("rel_bbox")
        if not isinstance(rel_bbox, Sequence) or isinstance(rel_bbox, (str, bytes)) or len(rel_bbox) != 4:
            continue
        try:
            left, top, width, height = (float(value) for value in rel_bbox)
        except (TypeError, ValueError):
            continue
        if left <= x <= left + width and top <= y <= top + height:
            return True
    return False


def _ground_truth_action(raw: object, *, index: int) -> tuple[str, Any, Mapping[str, Any]]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("type"), str):
        raise ValueError(f"ground_truth_actions[{index}] is malformed")
    action_type = raw["type"]
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"ground_truth_actions[{index}].params is malformed")
    if action_type in {"click", "doubleClick", "rightClick", "middleClick", "tripleClick", "moveTo", "dragTo"}:
        value = _action_position(raw, label=f"ground_truth_actions[{index}]")
    elif action_type in {"write", "press", "hotkey", "scroll", "hscroll", "wait", "terminate"}:
        if action_type == "write":
            value = params.get("text", params.get("content", ""))
        elif action_type in {"press", "hotkey"}:
            value = params.get("keys", [])
        elif action_type in {"scroll", "hscroll"}:
            value = params.get("amount", params.get("pixels", params.get("direction")))
        elif action_type == "wait":
            value = params.get("seconds", params.get("duration", 0))
        else:
            value = params.get("status")
    else:
        raise ValueError(f"ground_truth_actions[{index}] uses unsupported action {action_type!r}")
    return action_type, value, params


def _prediction_action(raw: object, *, index: int) -> tuple[str, Any]:
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        action_type = str(raw[0])
        return action_type, raw[1]
    if not isinstance(raw, Mapping):
        raise TypeError(f"predicted_actions[{index}] must be a mapping or pair")
    if isinstance(raw.get("name"), str):
        action_type = _TOOL_TO_ACTION.get(raw["name"])
        arguments = raw.get("arguments", {})
        if action_type is None or not isinstance(arguments, Mapping):
            raise ValueError(f"predicted_actions[{index}] has an unsupported tool call")
        if action_type in {"click", "doubleClick", "rightClick", "middleClick", "tripleClick", "moveTo", "dragTo"}:
            value = (arguments.get("x"), arguments.get("y"))
        elif action_type == "write":
            value = arguments.get("text", "")
        elif action_type in {"press", "hotkey"}:
            value = arguments.get("keys", [])
        elif action_type in {"scroll", "hscroll"}:
            value = arguments.get("clicks", arguments.get("amount", arguments.get("direction")))
        elif action_type == "terminate":
            value = arguments.get("status")
        else:
            value = arguments.get("seconds")
        return action_type, value
    if isinstance(raw.get("type"), str):
        action_type = raw["type"]
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"predicted_actions[{index}].params is malformed")
        if action_type in {"click", "doubleClick", "rightClick", "middleClick", "tripleClick", "moveTo", "dragTo"}:
            position = params.get("position", {})
            if not isinstance(position, Mapping):
                raise ValueError(f"predicted_actions[{index}].params.position is malformed")
            value = (position.get("x"), position.get("y"))
        elif action_type == "write":
            value = params.get("text", params.get("content", ""))
        elif action_type in {"press", "hotkey"}:
            value = params.get("keys", [])
        elif action_type in {"scroll", "hscroll"}:
            value = params.get("amount", params.get("pixels", params.get("direction")))
        else:
            value = params.get("status")
        return action_type, value
    raise ValueError(f"predicted_actions[{index}] has no action name")


def _flatten_ground_truth(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(record.get("ground_truth_actions"), list):
        return [item for item in record["ground_truth_actions"] if isinstance(item, Mapping)]
    steps = record.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise ValueError("AgentNet record must contain steps or ground_truth_actions")
    actions: list[Mapping[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"steps[{index}] must be an object")
        raw_actions = step.get("ground_truth_actions")
        if not isinstance(raw_actions, list):
            raise ValueError("steps must contain ground_truth_actions for official scoring")
        actions.extend(item for item in raw_actions if isinstance(item, Mapping))
    return actions


def _merge_ground_truth(actions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    merged: list[Mapping[str, Any]] = []
    index = 0
    while index < len(actions):
        current = actions[index]
        if index + 1 < len(actions):
            following = actions[index + 1]
            current_type = str(current.get("type", "")).casefold()
            following_type = str(following.get("type", "")).casefold()
            following_params = following.get("params", {})
            keys = following_params.get("keys", []) if isinstance(following_params, Mapping) else []
            if (
                current_type == "write"
                and following_type in {"press", "hotkey"}
                and _normalize_keys(keys) in [["enter"], ["return"]]
            ):
                current_params = current.get("params", {})
                if not isinstance(current_params, Mapping):
                    raise ValueError("write action params must be an object")
                copied = dict(current)
                copied["params"] = {
                    **current_params,
                    "text": str(current_params.get("text", current_params.get("content", ""))) + "\n",
                }
                merged.append(copied)
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def _merge_predictions(actions: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    merged: list[tuple[str, Any]] = []
    index = 0
    while index < len(actions):
        current = actions[index]
        if index + 1 < len(actions):
            following = actions[index + 1]
            if (
                current[0].casefold() == "write"
                and following[0].casefold() == "press"
                and _normalize_keys(following[1]) in [["enter"], ["return"]]
            ):
                merged.append(("write", str(current[1]) + "\n"))
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def _score_value(action_type: str, predicted: Any, ground_truth: Mapping[str, Any]) -> float:
    if action_type in {"click", "doubleClick", "rightClick", "middleClick", "tripleClick", "moveTo", "dragTo"}:
        px, py = _finite_pair(predicted, label="predicted coordinate")
        if _point_in_bboxes((px, py), ground_truth):
            return 1.0
        gx, gy = _action_position(ground_truth, label="ground truth")
        distance = math.hypot(px - gx, py - gy)
        return 1.0 if distance <= _COORD_THRESHOLD else math.exp(-_ALPHA * (distance - _COORD_THRESHOLD))
    if action_type == "write":
        expected = str(ground_truth.get("params", {}).get("text", ground_truth.get("params", {}).get("content", "")))
        actual = str(predicted)
        # Keep the public evaluator's ordering: strip outer whitespace before checking newline.
        # This intentionally preserves its behavior for a missing/extra trailing newline.
        expected = expected.casefold().strip()
        actual = actual.casefold().strip()
        expected_newline = expected.endswith("\n")
        actual_newline = actual.endswith("\n")
        expected = expected.rstrip("\n")
        actual = actual.rstrip("\n")
        maximum = max(len(expected), len(actual))
        similarity = 1.0 if maximum == 0 else 1.0 - _edit_distance(actual, expected) / maximum
        if expected_newline != actual_newline:
            similarity *= 0.9
        return 1.0 if similarity >= _WRITE_SIMILARITY_THRESHOLD else similarity / _WRITE_SIMILARITY_THRESHOLD
    if action_type in {"press", "hotkey"}:
        expected = ground_truth.get("params", {}).get("keys", [])
        return float(_normalize_keys(predicted) == _normalize_keys(expected))
    if action_type in {"scroll", "hscroll"}:
        params = ground_truth.get("params", {})
        expected = params.get("amount", params.get("pixels", params.get("direction")))
        expected_direction = _direction(expected)
        predicted_direction = _direction(predicted)
        if expected_direction and predicted_direction != expected_direction:
            return 0.0
        if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
            actual_abs, expected_abs = abs(float(predicted)), abs(float(expected))
            if actual_abs == expected_abs == 0:
                return 1.0
            if actual_abs == 0 or expected_abs == 0:
                return 0.0
            return min(actual_abs, expected_abs) / max(actual_abs, expected_abs)
        return float(predicted_direction == expected_direction)
    if action_type == "wait":
        try:
            return float(float(predicted) == float(ground_truth.get("params", {}).get("seconds", 0)))
        except (TypeError, ValueError):
            return 0.0
    if action_type == "terminate":
        return float(str(predicted).casefold() == str(ground_truth.get("params", {}).get("status")).casefold())
    return 0.0


def score_agentnet_actions(
    ground_truth_actions: Sequence[Mapping[str, Any]],
    predicted_actions: Sequence[object],
) -> dict[str, Any]:
    """Return the public AgentNetBench-compatible action score for one trajectory."""

    if not ground_truth_actions or not predicted_actions:
        return {
            "total": 0.0,
            "actions": {},
            "ground_truth_count": len(ground_truth_actions),
            "predicted_count": len(predicted_actions),
            "action_count_penalty": 1.0,
            "first_action_type_match": False,
            "claim_scope": "offline AgentNetBench-compatible proxy; not official leaderboard output",
        }
    merged_ground_truth = _merge_ground_truth(ground_truth_actions)
    truth = [_ground_truth_action(action, index=index) for index, action in enumerate(merged_ground_truth)]
    predictions = _merge_predictions(
        [_prediction_action(action, index=index) for index, action in enumerate(predicted_actions)]
    )
    # AgentNetBench penalizes over-prediction only; a short stream is scored as zero for the
    # missing suffix but does not receive an additional length penalty.
    penalty = min(1.0, len(truth) / len(predictions)) if len(predictions) > len(truth) else 1.0
    if truth[0][0].casefold() == "terminate":
        first_match = predictions[0][0].casefold() == "terminate" if predictions else False
        total = penalty * float(predictions[0][0].casefold() == "terminate" and predictions[0][1] == truth[0][1])
        return {
            "total": total,
            "actions": {"terminate": total},
            "ground_truth_count": len(truth),
            "predicted_count": len(predictions),
            "action_count_penalty": penalty,
            "first_action_type_match": first_match,
            "claim_scope": "offline AgentNetBench-compatible proxy; not official leaderboard output",
        }
    truth_types = [action[0].casefold() for action in truth]
    first_match = predictions[0][0].casefold() in truth_types
    if not first_match:
        return {
            "total": 0.0,
            "actions": {action_type: 0.0 for action_type in sorted(set(truth_types))},
            "ground_truth_count": len(truth),
            "predicted_count": len(predictions),
            "action_count_penalty": penalty,
            "first_action_type_match": False,
            "claim_scope": "offline AgentNetBench-compatible proxy; not official leaderboard output",
        }
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for index, (action_type, _, _) in enumerate(truth):
        normalized = action_type.casefold()
        # The public evaluator has a special termination branch only when termination is the
        # first action; a trailing termination marker is not included in its per-type average.
        if normalized == "terminate":
            continue
        counts[normalized] += 1
        if index >= len(predictions) or predictions[index][0].casefold() != normalized:
            continue
        scores[normalized] += _score_value(action_type, predictions[index][1], merged_ground_truth[index])
    for action_type, count in counts.items():
        scores[action_type] = scores[action_type] / count * penalty
    action_score = sum(scores.values()) / len(scores) if scores else 0.0
    total = action_score
    return {
        "total": total,
        "actions": dict(scores),
        "ground_truth_count": len(truth),
        "predicted_count": len(predictions),
        "action_count_penalty": penalty,
        "first_action_type_match": first_match,
        "claim_scope": "offline AgentNetBench-compatible proxy; not official leaderboard output",
    }


def score_agentnet_record(record: Mapping[str, Any], predicted_actions: Sequence[object]) -> dict[str, Any]:
    """Flatten an official AgentNet record and score a prediction stream."""

    return score_agentnet_actions(_flatten_ground_truth(record), predicted_actions)
