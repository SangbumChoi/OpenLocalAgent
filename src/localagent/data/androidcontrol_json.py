"""Adapter for the public, LLaMA-Factory-style AndroidControl JSON mirror.

The mirror contains one JSON object per screenshot/action pair.  This adapter deliberately keeps
the image reference as provenance only: the current LocalAgent checkpoint is text-only and must
not be credited with screenshot grounding.  The resulting rows are still useful for action
vocabulary and argument-format continuation, and the manifest records that visual input was
omitted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from localagent.agent.mobile_toolset import mobile_tools
from localagent.data.schema import Conversation, Message, Role, ToolCall


_ACTION_NAMES = {
    "click": "mobile_click",
    "long_press": "mobile_long_press",
    "scroll": "mobile_scroll",
    "open_app": "mobile_open_app",
    "input_text": "mobile_input_text",
    "navigate_home": "mobile_navigate_home",
    "navigate_back": "mobile_navigate_back",
    "press_enter": "mobile_press_enter",
    "wait": "mobile_wait",
}
_ACTION_FIELDS = {
    "mobile_click": frozenset(("x", "y")),
    "mobile_long_press": frozenset(("x", "y")),
    "mobile_scroll": frozenset(("direction",)),
    "mobile_open_app": frozenset(("app_name",)),
    "mobile_input_text": frozenset(("text",)),
    "mobile_navigate_home": frozenset(),
    "mobile_navigate_back": frozenset(),
    "mobile_press_enter": frozenset(("key",)),
    "mobile_wait": frozenset(),
}


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return " ".join(value.replace("<image>", "").split())


def _finite_number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _action(raw: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError("assistant action must be an object")
    action_type = _nonempty_text(raw.get("action_type"), "action_type").casefold()
    name = _ACTION_NAMES.get(action_type)
    if name is None:
        raise ValueError(f"unsupported AndroidControl action_type {action_type!r}")
    arguments = {str(key): value for key, value in raw.items() if key != "action_type"}
    expected = _ACTION_FIELDS[name]
    if set(arguments) != set(expected):
        raise ValueError(
            f"{action_type!r} arguments must contain exactly {sorted(expected)}; "
            f"got {sorted(arguments)}"
        )
    if name in {"mobile_click", "mobile_long_press"}:
        arguments = {key: _finite_number(value, key) for key, value in arguments.items()}
    elif name in {"mobile_scroll", "mobile_open_app", "mobile_input_text", "mobile_press_enter"}:
        for key, value in arguments.items():
            arguments[key] = _nonempty_text(value, key)
    return name, arguments


def json_row_to_conversation(
    raw: object,
    *,
    source_revision: str,
    split: str,
    row_index: int,
) -> Conversation:
    """Convert one public mirror row while making the missing visual input explicit."""

    if not isinstance(raw, Mapping):
        raise ValueError("AndroidControl JSON row must be an object")
    messages = raw.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("AndroidControl JSON row messages must be a sequence")
    user = next((item for item in messages if isinstance(item, Mapping) and item.get("role") == "user"), None)
    assistant = next(
        (item for item in reversed(messages) if isinstance(item, Mapping) and item.get("role") == "assistant"),
        None,
    )
    if user is None or assistant is None:
        raise ValueError("AndroidControl JSON row must contain user and assistant messages")
    instruction = _nonempty_text(user.get("content"), "user.content")
    content = assistant.get("content")
    if not isinstance(content, str):
        raise ValueError("assistant.content must be JSON text")
    try:
        name, arguments = _action(json.loads(content))
    except json.JSONDecodeError as error:
        raise ValueError("assistant.content is not valid JSON") from error
    image_refs = raw.get("images", [])
    if not isinstance(image_refs, Sequence) or isinstance(image_refs, (str, bytes)):
        raise ValueError("images must be a sequence when present")
    meta = {
        "source_family": "androidcontrol_json_mirror",
        "source_revision": _nonempty_text(source_revision, "source_revision"),
        "source_split": _nonempty_text(split, "split"),
        "source_row_index": row_index,
        "image_references": [str(value) for value in image_refs],
        "visual_input_omitted": True,
        "text_first": False,
        "grounding_evaluable": False,
    }
    tool_specs = mobile_tools()
    return Conversation(
        messages=[
            Message(
                role=Role.system,
                content="You are a mobile action parser. The screenshot is omitted; emit only the action JSON.",
            ),
            Message(
                role=Role.user,
                content=(
                    "Screenshot omitted for the text-only checkpoint. "
                    f"Instruction: {instruction}"
                ),
            ),
            Message(role=Role.assistant, tool_calls=[ToolCall(name=name, arguments=arguments)]),
            Message(role=Role.tool, tool_response="Action recorded; no device was changed."),
        ],
        tools=tool_specs,
        meta=meta,
    )


def canonical_action_from_conversation(conversation: Conversation) -> tuple[str, dict[str, Any]]:
    """Return the sole action in an adapted row for deterministic diagnostics."""

    calls = [call for message in conversation.messages for call in message.tool_calls]
    if len(calls) != 1:
        raise ValueError("AndroidControl JSON adapter rows must contain exactly one tool call")
    return calls[0].name, dict(calls[0].arguments)
