"""Dependency-light adapter for the official Android-in-the-Wild TFRecords.

AITW stores one GZIP TFRecord per interaction step rather than one record per episode.  This
module reconstructs complete episodes, projects the annotated UI text to a deterministic textual
observation, and maps the official action enum to the text-first mobile action bridge.  Screenshots
are deliberately not decoded or copied into the training rows.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from localagent.data.androidcontrol import (
    _bytes_list,
    _example_features,
    _float_list,
    _int_list,
    iter_gzip_tfrecords,
)
from localagent.data.realistic_adapters import normalize_mobile_row

_ACTION_TYPE = {
    3: "type",
    4: "dual_point",
    5: "navigate_back",
    6: "navigate_home",
    7: "press_enter",
    10: "task_complete",
    11: "task_impossible",
}
_STATUS_ACTIONS = frozenset({"task_complete", "task_impossible"})


def _first_text(features: Mapping[str, tuple[list[bytes], list[int], list[float]]], key: str) -> str:
    values = _bytes_list(features, key)
    if not values:
        raise ValueError(f"AITW example is missing text feature {key!r}")
    return values[0].decode("utf-8", errors="strict")


def _decode_episode_id(features: Mapping[str, tuple[list[bytes], list[int], list[float]]]) -> int:
    raw = _first_text(features, "episode_id")
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"AITW episode_id is not an integer: {raw!r}") from error


def _screen_text(features: Mapping[str, tuple[list[bytes], list[int], list[float]]]) -> str:
    texts = _bytes_list(features, "image/ui_annotations_text")
    types = _bytes_list(features, "image/ui_annotations_ui_types")
    positions = _float_list(features, "image/ui_annotations_positions")
    width = _int_list(features, "image/width")
    height = _int_list(features, "image/height")
    screen_width = float(width[0]) if width else 1.0
    screen_height = float(height[0]) if height else 1.0
    activity = _bytes_list(features, "current_activity")
    lines = [
        "SCREEN activity=" + (activity[0].decode("utf-8", errors="replace") if activity else ""),
        f"SCREEN size=({int(screen_width)}, {int(screen_height)})",
    ]
    count = max(len(texts), len(types), len(positions) // 4)
    for index in range(min(count, 160)):
        label = texts[index].decode("utf-8", errors="replace").strip() if index < len(texts) else ""
        ui_type = types[index].decode("utf-8", errors="replace").strip() if index < len(types) else ""
        start = index * 4
        if start + 4 <= len(positions):
            # Official order is (y, x, height, width), normalized to the screen dimensions.
            y, x, box_h, box_w = positions[start : start + 4]
            bounds = (
                round(x * screen_width, 1),
                round(y * screen_height, 1),
                round((x + box_w) * screen_width, 1),
                round((y + box_h) * screen_height, 1),
            )
        else:
            bounds = None
        if label or ui_type or bounds is not None:
            lines.append(
                f"UI[{index}] type={ui_type!r} text={label!r} bounds={bounds!r}"
            )
    return "\n".join(lines)[:12_000]


def _pixel_coordinate(value: float, extent: float) -> float:
    if not math.isfinite(value):
        raise ValueError("AITW action coordinate is not finite")
    # AITW publishes normalized y/x coordinates.  Keep this tolerant for mirrors that expose
    # already-pixel coordinates while retaining deterministic numeric arguments.
    return round(value * extent, 3) if abs(value) <= 1.5 else round(value, 3)


def _action(
    features: Mapping[str, tuple[list[bytes], list[int], list[float]]],
) -> dict[str, Any] | None:
    action_values = _int_list(features, "results/action_type")
    if not action_values:
        raise ValueError("AITW example has no results/action_type")
    action_name = _ACTION_TYPE.get(int(action_values[0]))
    if action_name is None:
        raise ValueError(f"unsupported AITW action enum {action_values[0]}")
    if action_name in _STATUS_ACTIONS:
        return None
    if action_name == "type":
        return {"action_type": "input_text", "text": _first_text(features, "results/type_action")}
    if action_name == "navigate_back":
        return {"action_type": action_name}
    if action_name == "navigate_home":
        return {"action_type": action_name}
    if action_name == "press_enter":
        return {"action_type": action_name, "key": "ENTER"}
    touch = _float_list(features, "results/yx_touch")
    lift = _float_list(features, "results/yx_lift")
    width = _int_list(features, "image/width")
    height = _int_list(features, "image/height")
    if len(touch) < 2 or len(lift) < 2 or not width or not height:
        raise ValueError("AITW dual-point action is missing coordinates or screen dimensions")
    screen_width, screen_height = float(width[0]), float(height[0])
    start_y, start_x = touch[:2]
    end_y, end_x = lift[:2]
    start = (_pixel_coordinate(start_x, screen_width), _pixel_coordinate(start_y, screen_height))
    end = (_pixel_coordinate(end_x, screen_width), _pixel_coordinate(end_y, screen_height))
    distance = math.hypot(end_x - start_x, end_y - start_y)
    if distance <= 0.04:
        return {"action_type": "click", "x": start[0], "y": start[1]}
    return {
        "action_type": "swipe",
        "start_x": start[0],
        "start_y": start[1],
        "end_x": end[0],
        "end_y": end[1],
    }


def iter_aitw_episodes(
    path: str | Path,
    *,
    allow_truncated: bool = False,
    verify_crc: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield complete AITW episodes reconstructed from step-level TFRecords."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_gzip_tfrecords(
        path,
        allow_truncated=allow_truncated,
        verify_crc=verify_crc,
    ):
        features = _example_features(payload)
        episode_id = _decode_episode_id(features)
        step_values = _int_list(features, "step_id")
        length_values = _int_list(features, "episode_length")
        if not step_values or not length_values:
            raise ValueError(f"AITW episode {episode_id} has no step metadata")
        grouped[episode_id].append(
            {
                "episode_id": episode_id,
                "step_id": int(step_values[0]),
                "episode_length": int(length_values[0]),
                "goal": _first_text(features, "goal_info"),
                "screen_text": _screen_text(features),
                "action": _action(features),
            }
        )
    for episode_id in sorted(grouped):
        steps = sorted(grouped[episode_id], key=lambda row: int(row["step_id"]))
        expected = int(steps[0]["episode_length"])
        if len(steps) != expected or [int(step["step_id"]) for step in steps] != list(range(expected)):
            if allow_truncated:
                continue
            raise ValueError(
                f"AITW episode {episode_id} is incomplete: {len(steps)} records for {expected} steps"
            )
        yield {"episode_id": episode_id, "goal": steps[0]["goal"], "steps": steps}


def episode_to_intermediate(
    episode: Mapping[str, Any],
    *,
    source_revision: str,
    split: str,
) -> dict[str, Any]:
    """Convert one complete AITW episode to the audited mobile intermediate row."""

    episode_id = episode.get("episode_id")
    goal = episode.get("goal")
    raw_steps = episode.get("steps")
    if not isinstance(episode_id, int) or not isinstance(goal, str) or not isinstance(raw_steps, list):
        raise ValueError("AITW episode shape is invalid")
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"AITW episode {episode_id} step {index} is invalid")
        action = raw_step.get("action")
        if action is None:
            # Terminal status actions are not executable tool calls and are kept only in quality.
            continue
        next_observation = None
        if index + 1 < len(raw_steps):
            next_observation = raw_steps[index + 1].get("screen_text")
        steps.append(
            {
                "instruction": f"Continue the task: {goal}",
                "screen_text": str(raw_step.get("screen_text", "")),
                "next_observation": next_observation,
                "action": action,
            }
        )
    if not steps:
        raise ValueError(f"AITW episode {episode_id} has no executable actions")
    row = normalize_mobile_row(
        {"record_id": f"aitw:{episode_id}", "goal": goal, "steps": steps},
        family="android_in_the_wild",
        source_revision=source_revision,
    )
    row["quality"].update(
        {
            "source_episode_id": episode_id,
            "source_split": split,
            "source_episode_steps": len(raw_steps),
            "source_executable_steps": len(steps),
            "action_matching": "google_research_aitw_action_space_v1",
            "screen_projection": "aitw_annotations_text_v1",
        }
    )
    return row


def _jsonable_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(episode, sort_keys=True))
