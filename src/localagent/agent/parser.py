"""Parse `<tool_call>...</tool_call>` spans out of model output into ToolCalls (Phase 4/7).

Shared by the agent runtime and the AST evaluator. Tolerant of minor JSON noise. A real impl
should add light JSON repair; for now it does a strict-ish extraction.
"""

from __future__ import annotations

import json
import re

from localagent.data.schema import ToolCall

_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Return all well-formed tool calls in `text` (possibly empty = plain text / abstention)."""
    calls: list[ToolCall] = []
    for m in _CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append(ToolCall(name=obj["name"], arguments=obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            # TODO(phase-7): JSON repair before giving up
            continue
    return calls


def strip_tool_calls(text: str) -> str:
    """The user-facing text with tool-call spans removed."""
    return _CALL_RE.sub("", text).strip()
