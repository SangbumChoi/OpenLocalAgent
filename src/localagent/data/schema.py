"""Canonical interchange schema (APIGen/xLAM-style).

ONE format flows through: synth -> flywheel -> SFT/distill -> eval. One JSONL line = one
Conversation. Add a field here and every downstream stage sees it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass
class ToolSpec:
    """A callable tool's name + JSON-schema parameters (OpenAI-function style)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def normalized(self) -> tuple[str, str]:
        """Canonical (name, sorted-json-args) used by the AST evaluator for comparison."""
        return self.name, json.dumps(self.arguments, sort_keys=True)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant tool calls
    tool_response: str | None = None  # for role == tool


@dataclass
class Conversation:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    # flywheel/feedback (Airbnb AITL) — all optional, populated by the conversation store
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_enum_default)

    @classmethod
    def from_json(cls, line: str) -> "Conversation":
        raw = json.loads(line)
        tools = [ToolSpec(**t) for t in raw.get("tools", [])]
        msgs = []
        for m in raw["messages"]:
            calls = [ToolCall(**c) for c in m.get("tool_calls", [])]
            msgs.append(
                Message(
                    role=Role(m["role"]),
                    content=m.get("content", ""),
                    tool_calls=calls,
                    tool_response=m.get("tool_response"),
                )
            )
        return cls(messages=msgs, tools=tools, meta=raw.get("meta", {}))


def _enum_default(o: Any):
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"not serializable: {type(o)}")
