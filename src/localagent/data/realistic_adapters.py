"""Fail-closed normalization for text-first mobile-control records.

Upstream AndroidControl is distributed as GZIP TFRecords and AITW includes screenshots.  The
project deliberately keeps those format-specific decoders out of the training dependency set.
This module defines the audited intermediate row that a decoder must emit before handing records
to the existing ``localagent_v1`` adapter.  Screenshot-only rows are rejected because the current
WebGPU checkpoint has no vision encoder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

_FAMILIES = frozenset({"androidcontrol", "android_in_the_wild"})
_ACTION_NAMES = {
    "click": "mobile_click",
    "long_press": "mobile_long_press",
    "scroll": "mobile_scroll",
    "swipe": "mobile_swipe",
    "open_app": "mobile_open_app",
    "input_text": "mobile_input_text",
    "navigate_home": "mobile_navigate_home",
    "navigate_back": "mobile_navigate_back",
    "press_enter": "mobile_press_enter",
    "wait": "mobile_wait",
}
_ACTION_PROPERTIES: dict[str, dict[str, dict[str, str]]] = {
    "mobile_click": {"x": {"type": "number"}, "y": {"type": "number"}},
    "mobile_long_press": {"x": {"type": "number"}, "y": {"type": "number"}},
    "mobile_scroll": {"direction": {"type": "string"}},
    "mobile_swipe": {
        "start_x": {"type": "number"},
        "start_y": {"type": "number"},
        "end_x": {"type": "number"},
        "end_y": {"type": "number"},
    },
    "mobile_open_app": {"app_name": {"type": "string"}},
    "mobile_input_text": {"text": {"type": "string"}},
    "mobile_navigate_home": {},
    "mobile_navigate_back": {},
    "mobile_press_enter": {"key": {"type": "string"}},
    "mobile_wait": {"seconds": {"type": "number"}},
}
_ACTION_REQUIRED: dict[str, set[str]] = {
    name: set(properties) for name, properties in _ACTION_PROPERTIES.items()
}
# AndroidControl encodes ``wait`` as an action with no arguments.  Some downstream exports add
# an optional duration, so accept either representation without inventing a default duration.
_ACTION_REQUIRED["mobile_wait"] = set()


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _json_value(value: object, *, label: str) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label=f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        return {
            _text(key, label=f"{label}.key"): _json_value(item, label=f"{label}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"{label} contains unsupported value {type(value).__name__}")


def _observation(step: Mapping[str, Any], *, index: int) -> str:
    """Return a deterministic text projection; reject pixels without text evidence."""

    candidates = (step.get("accessibility_tree"), step.get("screen_text"), step.get("observation"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, Mapping) or isinstance(candidate, list):
            return json.dumps(_json_value(candidate, label=f"steps[{index}].observation"), sort_keys=True)
    raise ValueError(
        f"steps[{index}] has no text/accessibility projection; screenshot-only rows cannot train "
        "the text-first WebGPU checkpoint"
    )


def _leaf_slots(value: Any, *, prefix: str, slots: dict[str, list[Any]]) -> None:
    if isinstance(value, Mapping):
        for key, child in sorted(value.items(), key=lambda pair: str(pair[0])):
            _leaf_slots(child, prefix=f"{prefix}.{key}", slots=slots)
    elif isinstance(value, list):
        for child in value:
            _leaf_slots(child, prefix=prefix, slots=slots)
    elif value is not None:
        slots.setdefault(prefix, []).append(_json_value(value, label=prefix))


def _tool(name: str) -> dict[str, Any]:
    properties = _ACTION_PROPERTIES[name]
    return {
        "name": name,
        "description": f"Execute the {name.removeprefix('mobile_').replace('_', ' ')} action.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": sorted(_ACTION_REQUIRED[name]),
            "additionalProperties": False,
        },
    }


def normalize_mobile_row(
    raw: object,
    *,
    family: str,
    source_revision: str,
) -> dict[str, Any]:
    """Convert one decoder-produced mobile row into a ``localagent_v1`` record.

    Required intermediate shape::

        {"record_id": str, "goal": str, "steps": [{"instruction": str,
        "accessibility_tree"|"screen_text": str, "action": {"action_type": str, ...}}]}

    ``next_observation`` is optional; if absent the tool response explicitly states that the
    source export omitted it rather than fabricating a successful environment result.
    """

    if family not in _FAMILIES:
        raise ValueError(f"unsupported mobile family {family!r}")
    row = raw if isinstance(raw, Mapping) else None
    if row is None:
        raise ValueError("mobile row must be a mapping")
    record_id = _text(row.get("record_id"), label="record_id")
    goal = _text(row.get("goal"), label="goal")
    steps = row.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise ValueError("steps must be a non-empty sequence")

    messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
    capabilities: set[str] = set()
    slot_values: dict[str, list[Any]] = {}
    tool_names: set[str] = set()
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"steps[{index}] must be a mapping")
        instruction = _text(raw_step.get("instruction", goal), label=f"steps[{index}].instruction")
        action = raw_step.get("action")
        if not isinstance(action, Mapping):
            raise ValueError(f"steps[{index}].action must be a mapping")
        action_type = _text(action.get("action_type"), label=f"steps[{index}].action.action_type")
        name = _ACTION_NAMES.get(action_type.casefold())
        if name is None:
            raise ValueError(f"steps[{index}] has unsupported action_type {action_type!r}")
        arguments = {
            str(key): _json_value(value, label=f"steps[{index}].action.{key}")
            for key, value in sorted(action.items())
            if key != "action_type"
        }
        expected = _ACTION_REQUIRED[name]
        allowed = set(_ACTION_PROPERTIES[name])
        if not expected.issubset(arguments) or not set(arguments).issubset(allowed):
            raise ValueError(
                f"steps[{index}] arguments for {action_type!r} must include {sorted(expected)} "
                f"and be limited to {sorted(allowed)}"
            )
        observation = _observation(raw_step, index=index)
        messages.append(
            {
                "role": "user",
                # Keep the actionable instruction at the right edge of the context. The model
                # applies left truncation at its fixed WebGPU window, while accessibility/screen
                # projections can be much longer than one window.
                "content": f"Current observation: {observation}\nStep {index + 1} instruction: {instruction}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": name, "arguments": arguments}],
            }
        )
        next_observation = raw_step.get("next_observation")
        if next_observation is None:
            response = "Action recorded; the source export omitted the post-action observation."
        elif isinstance(next_observation, str) and next_observation.strip():
            response = f"Post-action observation: {next_observation.strip()}"
        elif isinstance(next_observation, (Mapping, list)):
            response = "Post-action observation: " + json.dumps(
                _json_value(next_observation, label=f"steps[{index}].next_observation"),
                sort_keys=True,
            )
        else:
            raise ValueError(f"steps[{index}].next_observation has unsupported type")
        messages.append({"role": "tool", "tool_response": response})
        capabilities.add(name)
        tool_names.add(name)
        _leaf_slots(arguments, prefix=f"mobile.{action_type.casefold()}", slots=slot_values)

    messages.append({"role": "assistant", "content": "The mobile action trajectory was recorded."})
    return {
        "record_id": record_id,
        "domain": "mobile_ui",
        "behavior": "action",
        "capabilities": sorted(capabilities),
        "slot_values": {key: values for key, values in sorted(slot_values.items())},
        "quality": {
            "source_family": family,
            "source_revision": _text(source_revision, label="source_revision"),
            "normalization": "realistic_mobile_intermediate_v1",
            "text_first": True,
            "post_action_observation_present": any(
                "next_observation" in step for step in steps if isinstance(step, Mapping)
            ),
        },
        "tools": [_tool(name) for name in sorted(tool_names)],
        "messages": messages,
    }
