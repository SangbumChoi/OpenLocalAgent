#!/usr/bin/env python3
"""Normalize audited MCPMark trajectory logs into redacted Conversation JSONL.

This adapter is intentionally conservative.  It keeps user requests and structured tool-call
arguments, but replaces every tool result and assistant free-text response with a fixed marker and
redacts absolute workspace paths.  The resulting rows can teach multi-turn tool sequencing without
silently copying third-party documents or hidden reasoning into SFT data.  The output metadata
binds every source file by SHA-256 and records the redaction policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec


DATASET = "Jakumetsu/mcpmark-trajectory-log"
DEFAULT_REVISION = "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
_PATH_KEYS = frozenset({"path", "paths", "source", "destination", "directory", "root"})
_SAFE_SUFFIX = re.compile(r"[^A-Za-z0-9._/-]+")


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _parse_arguments(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        # MCPMark emits an empty string for zero-argument calls in some public traces.
        # Normalize that equivalent representation to the canonical empty object.
        if not value.strip():
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _redact_string(value: str, *, key: str, max_chars: int) -> str:
    if key in _PATH_KEYS and value.startswith("/"):
        suffix = "/".join(value.rstrip("/").split("/")[-3:])
        value = f"<workspace>/{_SAFE_SUFFIX.sub('_', suffix)}"
    if len(value) > max_chars:
        value = value[:max_chars].rstrip() + " …[argument truncated]"
    return value


def _redact_value(value: Any, *, key: str, max_chars: int) -> Any:
    if isinstance(value, str):
        return _redact_string(value, key=key, max_chars=max_chars)
    if isinstance(value, list):
        return [_redact_value(item, key=key, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        return {
            str(child_key): _redact_value(child_value, key=str(child_key), max_chars=max_chars)
            for child_key, child_value in value.items()
        }
    return value


def _schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _tool_specs(calls: list[tuple[str, dict[str, Any]]]) -> list[ToolSpec]:
    observations: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, arguments in calls:
        # Keep zero-argument tools in the catalog; otherwise their assistant calls would refer to
        # a tool absent from the canonical Conversation schema.
        observations[name]
        for key, value in arguments.items():
            observations[name][str(key)].append(_schema_type(value))
    specs: list[ToolSpec] = []
    for name in sorted(observations):
        properties = {
            key: {"type": sorted(set(types))[0]}
            for key, types in sorted(observations[name].items())
        }
        specs.append(
            ToolSpec(
                name=name,
                description="Schema inferred from a redacted public MCPMark trajectory.",
                parameters={
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": True,
                },
            )
        )
    return specs


def _load_events(path: Path, *, max_argument_chars: int) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read trajectory {path}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"trajectory must be a non-empty event list: {path}")
    calls: dict[str, str] = {}
    outputs: set[str] = set()
    parsed: list[dict[str, Any]] = []
    call_rows: list[tuple[str, dict[str, Any]]] = []
    for index, event in enumerate(raw):
        if not isinstance(event, Mapping):
            raise ValueError(f"event {index} must be an object: {path}")
        event_type = event.get("type")
        if event_type is None and event.get("role") in {"user", "assistant"}:
            event_type = "message"
        if event_type == "function_call":
            name = event.get("name")
            call_id = event.get("call_id")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"event {index} has invalid function name: {path}")
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ValueError(f"event {index} has invalid or duplicate call_id: {path}")
            arguments = _parse_arguments(event.get("arguments", {}), label=f"event {index}")
            redacted = _redact_value(arguments, key="", max_chars=max_argument_chars)
            calls[call_id] = name
            call_rows.append((name, redacted))
            parsed.append({"type": "function_call", "name": name, "call_id": call_id, "arguments": redacted})
        elif event_type == "function_call_output":
            call_id = event.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in outputs:
                raise ValueError(f"event {index} has invalid or duplicate output call_id: {path}")
            outputs.add(call_id)
            parsed.append({"type": "function_call_output", "call_id": call_id})
        elif event_type == "message":
            role = event.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError(f"event {index} has unsupported message role: {path}")
            parsed.append({"type": "message", "role": role, "content": event.get("content", "")})
        else:
            raise ValueError(f"event {index} has unsupported type {event_type!r}: {path}")
    if set(calls) != outputs:
        raise ValueError(f"unpaired MCPMark calls in {path}")
    if not any(event["type"] == "message" and event.get("role") == "user" for event in parsed):
        raise ValueError(f"trajectory has no user message: {path}")
    return parsed, call_rows


def normalize(
    inputs: list[Path],
    output: Path,
    metadata_output: Path,
    *,
    revision: str = DEFAULT_REVISION,
    max_prompt_chars: int = 4000,
    max_argument_chars: int = 512,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input trajectory is required")
    if max_prompt_chars < 1 or max_argument_chars < 1:
        raise ValueError("redaction limits must be positive")
    rows: list[Conversation] = []
    sources: list[dict[str, Any]] = []
    total_calls = 0
    for path in inputs:
        events, calls = _load_events(path, max_argument_chars=max_argument_chars)
        tool_specs = _tool_specs(calls)
        messages: list[Message] = []
        for event in events:
            if event["type"] == "message":
                if event["role"] == "user":
                    content = event.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
                        )
                    content = str(content)[:max_prompt_chars]
                    messages.append(Message(role=Role.user, content=content))
                else:
                    messages.append(
                        Message(role=Role.assistant, content="[assistant text redacted]")
                    )
            elif event["type"] == "function_call":
                messages.append(
                    Message(
                        role=Role.assistant,
                        tool_calls=[
                            ToolCall(name=event["name"], arguments=event["arguments"])
                        ],
                    )
                )
            else:
                messages.append(
                    Message(
                        role=Role.tool,
                        tool_response="[MCP tool output redacted for content audit]",
                    )
                )
        source_identity = file_identity(path)
        parent_id = source_identity["sha256"][:16]
        rows.append(
            Conversation(
                messages=messages,
                tools=tool_specs,
                meta={
                    "kind": "mcpmark_redacted_trajectory_v1",
                    "parent_record_id": parent_id,
                    "source_dataset": DATASET,
                    "source_revision": revision,
                    "source_path_sha256": source_identity["sha256"],
                    "tool_outputs_redacted": True,
                    "assistant_text_redacted": True,
                    "absolute_paths_redacted": True,
                    "training_content_policy": "redacted_tool_trace_only",
                },
            )
        )
        total_calls += len(calls)
        sources.append({**source_identity, "tool_calls": len(calls), "unique_tools": sorted({name for name, _ in calls})})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_json() + "\n")
    metadata = {
        "kind": "localagent_mcpmark_redacted_trajectory_normalization",
        "schema_version": 1,
        "dataset": DATASET,
        "revision": revision,
        "inputs": sources,
        "rows": len(rows),
        "tool_calls": total_calls,
        "output": file_identity(output),
        "redaction": {
            "tool_outputs": "fixed_marker",
            "assistant_free_text": "fixed_marker",
            "absolute_paths": "workspace_suffix_only",
            "max_prompt_chars": max_prompt_chars,
            "max_argument_chars": max_argument_chars,
        },
        "training_used": True,
        "claim_boundary": (
            "Redacted text-first continuation rows derived from public MCPMark trajectories. "
            "Tool outputs and assistant free text were excluded; this is not an official MCPMark "
            "score or evidence of live MCP execution. The source license covers the published logs, "
            "but downstream users must audit any retained user/tool argument content separately."
        ),
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-prompt-chars", type=int, default=4000)
    parser.add_argument("--max-argument-chars", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or args.metadata_output.exists():
        raise SystemExit("refusing to overwrite normalization outputs")
    report = normalize(
        args.input,
        args.output,
        args.metadata_output,
        revision=args.revision,
        max_prompt_chars=args.max_prompt_chars,
        max_argument_chars=args.max_argument_chars,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
