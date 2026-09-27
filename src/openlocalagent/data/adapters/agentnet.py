"""Fail-closed normalization for the public OpenCUA AgentNet trajectories.

AgentNet is a desktop computer-use dataset.  The public export has appeared in two compatible
JSON shapes: the original ``steps``/``ground_truth_actions`` records used by AgentNetBench and the
HF JSONL ``traj`` records whose ``value`` object contains ``observation`` and ``code``.  This
module accepts both shapes and emits the project's audited ``localagent_v1`` interchange.

The adapter deliberately preserves the low-level coordinate action space as ``agentnet_*`` tools.
Those tools are useful for offline trajectory scoring, but are not silently mapped to the
text-grounded WebGPU ``click``/``type_text`` tools.  A screenshot or accessibility bridge is
required before an AgentNet record can become a deployable WebGPU example.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

AGENTNET_REVISION = "d76ee50a63fad81cfdbe576416757d7c2091ed50"
AGENTNET_URL = "https://huggingface.co/datasets/xlangai/AgentNet"
AGENTNET_LICENSE = "MIT"

_PY_AUTO_GUI_ACTIONS = frozenset(
    {
        "click",
        "doubleClick",
        "rightClick",
        "middleClick",
        "moveTo",
        "dragTo",
        "scroll",
        "hscroll",
        "write",
        "press",
        "hotkey",
    }
)
_COMPUTER_ACTIONS = frozenset({"terminate", "wait", "triple_click"})
_ACTION_TO_TOOL = {
    "click": "agentnet_click",
    "doubleClick": "agentnet_double_click",
    "rightClick": "agentnet_right_click",
    "middleClick": "agentnet_middle_click",
    "moveTo": "agentnet_move_cursor",
    "dragTo": "agentnet_drag",
    "scroll": "agentnet_scroll",
    "hscroll": "agentnet_hscroll",
    "write": "agentnet_type_text",
    "press": "agentnet_key_press",
    "hotkey": "agentnet_hotkey",
    "tripleClick": "agentnet_triple_click",
    "wait": "agentnet_wait",
    "terminate": "agentnet_terminate",
}


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label=f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        return {
            _text(key, label=f"{label}.key"): _json_value(child, label=f"{label}.{key}")
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"{label} contains unsupported value {type(value).__name__}")


def _literal(node: ast.AST, *, label: str) -> Any:
    """Evaluate only literal AST nodes; never execute arbitrary AgentNet code."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item, label=label) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _literal(key, label=label): _literal(value, label=label)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand, label=label)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{label} has an invalid unary literal")
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f"{label} contains unsupported expression {type(node).__name__}")


def _call_arguments(call: ast.Call, *, label: str) -> dict[str, Any]:
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError(f"{label} uses **kwargs, which are not supported")
    values = [_literal(argument, label=label) for argument in call.args]
    keyword_values = {keyword.arg: _literal(keyword.value, label=label) for keyword in call.keywords}
    name = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    positional_names = {
        "click": ("x", "y", "clicks", "interval", "button"),
        "doubleClick": ("x", "y", "interval", "button"),
        "rightClick": ("x", "y"),
        "middleClick": ("x", "y"),
        "moveTo": ("x", "y", "duration"),
        "dragTo": ("x", "y", "duration", "button"),
        "scroll": ("clicks", "x", "y"),
        "hscroll": ("clicks", "x", "y"),
        "write": ("message", "interval"),
        "press": ("keys", "presses", "interval"),
        "hotkey": ("args", "interval"),
        "triple_click": ("x", "y"),
        "wait": ("seconds",),
        "terminate": ("status",),
    }.get(name, ())
    result = {
        key: value
        for key, value in zip(positional_names, values)
    }
    result.update(keyword_values)
    if name in {"press", "hotkey"} and values and "args" not in result:
        # The official serializer may use positional strings for keyboard actions.
        result["keys"] = values if name == "hotkey" else values[0]
    if name == "write" and "message" in result:
        result["text"] = result.pop("message")
    if name == "press" and "keys" in result and not isinstance(result["keys"], list):
        result["keys"] = [result["keys"]]
    if name == "hotkey":
        if values and not keyword_values:
            result = {"keys": values}
        elif "args" in result:
            result["keys"] = result.pop("args")
    return {
        str(key): _json_value(value, label=f"{label}.{key}")
        for key, value in sorted(result.items())
        if value is not None
    }


def parse_pyautogui_actions(source: str, *, label: str = "action") -> list[dict[str, Any]]:
    """Parse literal ``pyautogui`` calls into canonical AgentNet action dictionaries."""

    text = _text(source, label=label)
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise ValueError(f"{label} is not valid Python action code") from error
    actions: list[dict[str, Any]] = []
    for statement in tree.body:
        expression = statement.value if isinstance(statement, ast.Expr) else None
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Attribute):
            raise ValueError(f"{label} contains a non-call statement")
        target = expression.func.value
        if not isinstance(target, ast.Name) or target.id not in {"pyautogui", "computer"}:
            raise ValueError(f"{label} is not a supported computer-use call")
        action_type = expression.func.attr
        if target.id == "pyautogui" and action_type not in _PY_AUTO_GUI_ACTIONS:
            raise ValueError(f"{label} uses unsupported pyautogui action {action_type!r}")
        if target.id == "computer" and action_type not in _COMPUTER_ACTIONS:
            raise ValueError(f"{label} uses unsupported computer action {action_type!r}")
        normalized_type = "tripleClick" if action_type == "triple_click" else action_type
        actions.append(
            {
                "action_type": normalized_type,
                "arguments": _call_arguments(expression, label=f"{label}.{action_type}"),
            }
        )
    if not actions:
        raise ValueError(f"{label} contains no pyautogui calls")
    return actions


def _ground_truth_action(raw: object, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be an object")
    action_type = _text(raw.get("type"), label=f"{label}.type")
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError(f"{label}.params must be an object")
    normalized = action_type
    if action_type in {"click", "doubleClick", "rightClick", "middleClick", "moveTo", "dragTo", "tripleClick"}:
        position = params.get("position")
        if not isinstance(position, Mapping):
            raise ValueError(f"{label}.params.position is required")
        arguments = {"x": position.get("x"), "y": position.get("y")}
        if action_type == "dragTo":
            normalized = "dragTo"
    elif action_type in {"write", "press", "hotkey"}:
        if action_type == "write":
            arguments = {"text": params.get("text", params.get("content"))}
        else:
            arguments = {"keys": params.get("keys", [])}
    elif action_type in {"scroll", "hscroll"}:
        arguments = {"clicks": params.get("amount", params.get("pixels", params.get("direction")))}
    elif action_type == "terminate":
        arguments = {"status": params.get("status", "success")}
    elif action_type == "wait":
        arguments = {"seconds": params.get("seconds", params.get("duration", 0))}
    else:
        raise ValueError(f"{label} uses unsupported AgentNet action type {action_type!r}")
    return {
        "action_type": normalized,
        "arguments": {
            str(key): _json_value(value, label=f"{label}.params.{key}")
            for key, value in sorted(arguments.items())
            if value is not None
        },
    }


def _step_observation(step: Mapping[str, Any], *, label: str) -> str:
    inner = step.get("inner_monologue")
    value = step.get("value") if isinstance(step.get("value"), Mapping) else None
    candidates = [
        inner.get("observation") if isinstance(inner, Mapping) else None,
        step.get("observation"),
        value.get("observation") if value is not None else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError(
        f"{label} has no textual observation; screenshot-only AgentNet rows cannot train the "
        "text-first WebGPU checkpoint"
    )


def _step_actions(step: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    ground_truth = step.get("ground_truth_actions")
    if isinstance(ground_truth, list):
        if not ground_truth:
            raise ValueError(f"{label}.ground_truth_actions is empty")
        return [
            _ground_truth_action(action, label=f"{label}.ground_truth_actions[{index}]")
            for index, action in enumerate(ground_truth)
        ]
    value = step.get("value") if isinstance(step.get("value"), Mapping) else step
    code = value.get("code") if isinstance(value, Mapping) else None
    action = value.get("action") if isinstance(value, Mapping) else None
    source = code if isinstance(code, str) and code.strip() else action
    if not isinstance(source, str):
        raise ValueError(f"{label} has no supported action code")
    return parse_pyautogui_actions(source, label=f"{label}.action")


def _tool(action_type: str) -> dict[str, Any]:
    properties: dict[str, dict[str, str]] = {
        "click": {"x": {"type": "number"}, "y": {"type": "number"}},
        "doubleClick": {"x": {"type": "number"}, "y": {"type": "number"}, "interval": {"type": "number"}, "button": {"type": "string"}},
        "rightClick": {"x": {"type": "number"}, "y": {"type": "number"}},
        "middleClick": {"x": {"type": "number"}, "y": {"type": "number"}},
        "moveTo": {"x": {"type": "number"}, "y": {"type": "number"}, "duration": {"type": "number"}},
        "dragTo": {"x": {"type": "number"}, "y": {"type": "number"}, "duration": {"type": "number"}, "button": {"type": "string"}},
        "scroll": {"clicks": {}, "x": {"type": "number"}, "y": {"type": "number"}},
        "hscroll": {"clicks": {}, "x": {"type": "number"}, "y": {"type": "number"}},
        "write": {"text": {"type": "string"}, "interval": {"type": "number"}},
        "press": {"keys": {}, "presses": {}, "interval": {"type": "number"}},
        "hotkey": {"keys": {}, "interval": {"type": "number"}},
        "tripleClick": {"x": {"type": "number"}, "y": {"type": "number"}},
        "wait": {"seconds": {"type": "number"}},
        "terminate": {"status": {"type": "string", "enum": ["success", "failure"]}},
    }[action_type]
    required = {
        "click": ["x", "y"],
        "doubleClick": ["x", "y"],
        "rightClick": ["x", "y"],
        "middleClick": ["x", "y"],
        "moveTo": ["x", "y"],
        "dragTo": ["x", "y"],
        "scroll": ["clicks"],
        "hscroll": ["clicks"],
        "write": ["text"],
        "press": ["keys"],
        "hotkey": ["keys"],
        "tripleClick": ["x", "y"],
        "wait": ["seconds"],
        "terminate": ["status"],
    }[action_type]
    return {
        "name": _ACTION_TO_TOOL[action_type],
        "description": f"Replay the AgentNet {action_type} computer action offline.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def normalize_agentnet_record(
    raw: object,
    *,
    source_revision: str = AGENTNET_REVISION,
    split: str = "eval",
) -> dict[str, Any]:
    """Normalize one AgentNet record into the audited ``localagent_v1`` JSON shape."""

    if split != "eval":
        raise ValueError("AgentNet is evaluation-only; its official holdout cannot enter training")
    if not isinstance(raw, Mapping):
        raise TypeError("AgentNet record must be an object")
    record_id = _text(raw.get("task_id"), label="task_id")
    task = raw.get("user_task_description") or raw.get("instruction") or raw.get("natural_language_task")
    task = _text(task, label="task description")
    raw_steps = raw.get("steps")
    if raw_steps is None:
        raw_steps = raw.get("traj")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)) or not raw_steps:
        raise ValueError("AgentNet record must contain non-empty steps or traj")

    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    tool_map: dict[str, dict[str, Any]] = {}
    capabilities: set[str] = set()
    slot_values: dict[str, list[Any]] = {}
    action_count = 0
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise TypeError(f"AgentNet step {index} must be an object")
        observation = _step_observation(raw_step, label=f"steps[{index}]")
        actions = _step_actions(raw_step, label=f"steps[{index}]")
        instruction = raw_step.get("low_level_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            inner = raw_step.get("inner_monologue")
            instruction = inner.get("low_level_instruction") if isinstance(inner, Mapping) else None
        prompt = f"Current desktop observation: {observation}\nStep {index + 1} instruction: "
        prompt += instruction.strip() if isinstance(instruction, str) and instruction.strip() else task
        messages.append({"role": "user", "content": prompt})
        for action_index, action in enumerate(actions):
            action_type = action["action_type"]
            tool_name = _ACTION_TO_TOOL[action_type]
            arguments = action["arguments"]
            call = {"name": tool_name, "arguments": arguments}
            tool_map[action_type] = _tool(action_type)
            capabilities.add(tool_name)
            action_count += 1
            for key, value in arguments.items():
                slot_values.setdefault(f"{tool_name}.{key}", []).append(value)
            messages.append({"role": "assistant", "tool_calls": [call]})
            messages.append(
                {
                    "role": "tool",
                    "tool_response": f"AgentNet source action {index}:{action_index} recorded; no environment was executed.",
                }
            )

    if action_count < 1:
        raise ValueError(f"AgentNet record {record_id!r} has no executable actions")
    return {
        "record_id": f"agentnet:{record_id}",
        "domain": "computer_use",
        "behavior": "action",
        "capabilities": sorted(capabilities),
        "slot_values": {key: values for key, values in sorted(slot_values.items())},
        "quality": {
            "source_family": "agentnet",
            "source_revision": _text(source_revision, label="source_revision"),
            "source_split": split,
            "normalization": "agentnet_trajectory_v1",
            "text_first": True,
            "vision_available_but_not_consumed": True,
            "environment_executed": False,
            "action_count": action_count,
        },
        "tools": [tool_map[key] for key in sorted(tool_map)],
        "messages": messages,
    }
