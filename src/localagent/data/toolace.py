"""Fail-closed projection of the public ToolACE function-calling snapshot.

ToolACE is a large, synthetic function-calling corpus whose rows mix canonical calls with
deliberately difficult or non-canonical call spellings.  This adapter keeps only the first
assistant call that can be parsed as the documented ``[name(key=value), ...]`` form.  It emits
the project's canonical :class:`~localagent.data.schema.Conversation` interchange and records
the projection boundary explicitly; it does not claim to reproduce the full ToolACE benchmark
or its multi-turn training recipe.

Acquisition is intentionally separate.  The caller must provide the exact local bytes and
expected SHA-256, and the generated train/eval split is prompt-hash based rather than an
upstream official split.  This keeps the public source useful for a small-model continuation
without silently treating the source's synthetic data as a native environment evaluation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import self_hashed_manifest
from localagent.data.prompt_contract import assistant_training_turns
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec

TOOLACE_DATASET = "Team-ACE/ToolACE"
TOOLACE_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
TOOLACE_URL = "https://huggingface.co/datasets/Team-ACE/ToolACE"
TOOLACE_LICENSE = "Apache-2.0"
TOOLACE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
TOOLACE_ADAPTER_VERSION = "toolace-first-canonical-action-v1"
TOOLACE_MULTITURN_ADAPTER_VERSION = "toolace-multiturn-canonical-actions-v1"
TOOLACE_ACTION_HISTORY_ADAPTER_VERSION = "toolace-action-history-canonical-actions-v1"
TOOLACE_MANIFEST_KIND = "localagent_toolace_projection"
_SUPPORTED_SCHEMA_KEYS = {
    "type",
    "description",
    "enum",
    "format",
    "properties",
    "required",
    "additionalProperties",
    "items",
}


def _canonical_value(value: Any, *, label: str) -> Any:
    """Return finite JSON data, converting Python tuples to JSON arrays."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_canonical_value(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{label} object keys must be strings")
        return {key: _canonical_value(value[key], label=f"{label}.{key}") for key in sorted(value)}
    raise TypeError(f"{label} contains unsupported value {type(value).__name__}")


def _split_top_level(value: str) -> list[str]:
    """Split comma-separated calls while respecting strings and nested JSON/Python values."""

    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced ToolACE call delimiters")
        elif char == "," and depth == 0:
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    if quote is not None or depth != 0:
        raise ValueError("unterminated ToolACE call string")
    part = value[start:].strip()
    if part:
        parts.append(part)
    return parts


def parse_toolace_calls(value: object) -> tuple[ToolCall, ...]:
    """Parse the strict bracketed ToolACE call form, returning no calls on drift."""

    if not isinstance(value, str):
        return ()
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return ()
    inner = text[1:-1].strip()
    if not inner:
        return ()
    calls: list[ToolCall] = []
    try:
        segments = _split_top_level(inner)
    except ValueError:
        return ()
    for segment in segments:
        opening = segment.find("(")
        if opening <= 0 or not segment.endswith(")"):
            return ()
        name = segment[:opening].strip()
        arguments_text = segment[opening + 1 : -1]
        if not name or "\n" in name:
            return ()
        try:
            expression = ast.parse(f"_tool({arguments_text})", mode="eval").body
            if not isinstance(expression, ast.Call) or expression.args:
                return ()
            arguments: dict[str, Any] = {}
            for keyword in expression.keywords:
                if keyword.arg is None or keyword.arg in arguments:
                    return ()
                arguments[keyword.arg] = _canonical_value(
                    ast.literal_eval(keyword.value),
                    label=f"ToolACE call {name!r}.{keyword.arg}",
                )
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            return ()
        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)


def _sanitize_schema(value: object, *, force_object: bool = False) -> dict[str, Any]:
    """Keep the JSON-schema subset accepted by LocalAgent, falling back conservatively."""

    if not isinstance(value, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _SUPPORTED_SCHEMA_KEYS:
            continue
        if key == "type" and isinstance(item, str):
            result[key] = {
                "str": "string",
                "string": "string",
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "double": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "dict": "object",
                "object": "object",
                "list": "array",
                "array": "array",
            }.get(item.casefold(), item)
        elif key == "format" and isinstance(item, str):
            result[key] = item
        elif key == "description" and isinstance(item, str):
            result[key] = item
        elif key == "properties" and isinstance(item, dict):
            result[key] = {
                str(name): _sanitize_schema(schema, force_object=False)
                for name, schema in item.items()
            }
        elif key == "items" and isinstance(item, dict):
            result[key] = _sanitize_schema(item, force_object=False)
        elif key == "required" and isinstance(item, list):
            result[key] = [name for name in item if isinstance(name, str)]
        elif key == "enum" and isinstance(item, list):
            try:
                result[key] = _canonical_value(item, label="ToolACE schema enum")
            except (TypeError, ValueError):
                continue
        elif key == "additionalProperties" and isinstance(item, (bool, dict)):
            result[key] = (
                _sanitize_schema(item, force_object=False) if isinstance(item, dict) else item
            )
        elif isinstance(item, (str, bool, int, float)) or item is None:
            result[key] = item
    if force_object and result.get("type") != "object":
        result["type"] = "object"
    if force_object:
        result.setdefault("properties", {})
        result.setdefault("additionalProperties", True)
    return result


def _tool_spec(raw: object) -> ToolSpec | None:
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("function"), dict):
        raw = raw["function"]
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = raw.get("description", "")
    if not isinstance(description, str):
        description = str(description)
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    return ToolSpec(
        name=name.strip(),
        description=description,
        parameters=_sanitize_schema(parameters, force_object=True),
    )


def _system_tool_candidates(system: object) -> list[list[ToolSpec]]:
    if not isinstance(system, str):
        return []
    decoder = json.JSONDecoder()
    candidates: list[list[ToolSpec]] = []
    for index, char in enumerate(system):
        if char != "[":
            continue
        try:
            value, _end = decoder.raw_decode(system[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        tools = [tool for raw in value if (tool := _tool_spec(raw)) is not None]
        if tools:
            candidates.append(tools)
    return candidates


def _catalog_for_calls(system: object, calls: Iterable[ToolCall]) -> tuple[tuple[ToolSpec, ...], tuple[str, ...]]:
    call_names = {call.name for call in calls}
    candidates = _system_tool_candidates(system)
    selected = max(
        candidates,
        key=lambda tools: (len(call_names & {tool.name for tool in tools}), len(tools)),
        default=[],
    )
    by_name = {tool.name: tool for tool in selected}
    fallback: list[str] = []
    for name in sorted(call_names):
        if name not in by_name:
            by_name[name] = ToolSpec(
                name=name,
                description="ToolACE call observed; source schema unavailable in this projection.",
                parameters={"type": "object", "properties": {}, "additionalProperties": True},
            )
            fallback.append(name)
    return tuple(by_name[name] for name in sorted(by_name)), tuple(fallback)


def _first_action(raw: object) -> tuple[str, tuple[ToolCall, ...], int] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("conversations"), list):
        return None
    last_user: str | None = None
    for index, message in enumerate(raw["conversations"]):
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        if role == "user" and isinstance(message.get("value"), str) and message["value"].strip():
            last_user = message["value"].strip()
        elif role == "assistant":
            calls = parse_toolace_calls(message.get("value"))
            if calls and last_user is not None:
                return last_user, calls, index
    return None


def _all_action_calls(raw: object) -> list[tuple[int, tuple[ToolCall, ...]]]:
    """Return every strict assistant action, retaining its source message index."""

    if not isinstance(raw, dict) or not isinstance(raw.get("conversations"), list):
        return []
    actions: list[tuple[int, tuple[ToolCall, ...]]] = []
    for index, message in enumerate(raw["conversations"]):
        if not isinstance(message, dict) or message.get("from") != "assistant":
            continue
        calls = parse_toolace_calls(message.get("value"))
        if calls:
            actions.append((index, calls))
    return actions


def normalize_toolace_row(
    raw: object,
    *,
    record_index: int,
    split: str,
    source_sha256: str,
) -> Conversation | None:
    """Normalize one row into a first-action canonical Conversation."""

    action = _first_action(raw)
    if action is None:
        return None
    user, calls, assistant_index = action
    tools, fallback_tools = _catalog_for_calls(raw.get("system") if isinstance(raw, dict) else None, calls)
    parent_record_id = f"toolace-{record_index:05d}"
    conversation = Conversation(
        messages=[Message(role=Role.user, content=user), Message(role=Role.assistant, tool_calls=list(calls))],
        tools=list(tools),
        meta={
            "category": "toolace_function_calling",
            "group": "public_agent",
            "kind": "public_agent_trace",
            "public_data": True,
            "behavior": "action",
            "capabilities": sorted({call.name for call in calls}),
            "action_count": len(calls),
            "parent_record_id": parent_record_id,
            "split": split,
            "source_turn_index": assistant_index,
            "toolace_projection": TOOLACE_ADAPTER_VERSION,
            "schema_fallback_tools": list(fallback_tools),
            "slot_values": {},
            "slot_policy": "omitted; only parent-record and prompt disjointness are claimed",
            "derivation": "first_canonical_assistant_action_v1",
            "quality": {
                "source_conversation_turns": len(raw.get("conversations", [])) if isinstance(raw, dict) and isinstance(raw.get("conversations"), list) else None,
                "tool_response_omitted": True,
            },
            "provenance": {
                "dataset": TOOLACE_DATASET,
                "subset": "source_record_disjoint_train" if split == "train" else "source_record_disjoint_eval",
                "revision": TOOLACE_REVISION,
                "record_id": parent_record_id,
                "url": TOOLACE_URL,
                "license": TOOLACE_LICENSE,
                "file_sha256": source_sha256,
                "source_line": record_index + 1,
            },
            "verified": False,
            "rule_verified": True,
            "model_verified": False,
            "environment_executed": False,
            "verification_scope": "tool_catalog_schema_and_canonical_first_action_projection",
        },
    )
    assistant_training_turns(conversation)
    return conversation


def normalize_toolace_multiturn_row(
    raw: object,
    *,
    record_index: int,
    split: str,
    source_sha256: str,
    include_assistant_text: bool = True,
) -> Conversation | None:
    """Normalize all user, assistant, and tool turns while retaining strict action calls."""

    if not isinstance(raw, dict) or not isinstance(raw.get("conversations"), list):
        return None
    actions = _all_action_calls(raw)
    if not actions:
        return None
    calls = [call for _index, action_calls in actions for call in action_calls]
    tools, fallback_tools = _catalog_for_calls(raw.get("system"), calls)
    messages: list[Message] = []
    assistant_action_count = 0
    assistant_text_turns_omitted = 0
    tool_response_count = 0
    for index, raw_message in enumerate(raw["conversations"]):
        if not isinstance(raw_message, dict):
            raise ValueError(f"ToolACE conversation message {index} must be an object")
        role = raw_message.get("from")
        value = raw_message.get("value")
        if not isinstance(value, str):
            raise ValueError(f"ToolACE conversation message {index} value must be text")
        if role == "user":
            messages.append(Message(role=Role.user, content=value))
        elif role == "assistant":
            parsed = parse_toolace_calls(value)
            if parsed:
                messages.append(Message(role=Role.assistant, tool_calls=list(parsed)))
                assistant_action_count += 1
            elif include_assistant_text:
                messages.append(Message(role=Role.assistant, content=value))
            else:
                assistant_text_turns_omitted += 1
        elif role == "tool":
            messages.append(Message(role=Role.tool, tool_response=value))
            tool_response_count += 1
        else:
            raise ValueError(f"ToolACE conversation message {index} role is unsupported")
    parent_record_id = f"toolace-{record_index:05d}"
    conversation = Conversation(
        messages=messages,
        tools=list(tools),
        meta={
            "category": "toolace_function_calling",
            "group": "public_agent",
            "kind": "public_agent_trace",
            "public_data": True,
            "behavior": "stateful_action",
            "capabilities": sorted({call.name for call in calls}),
            "action_count": len(calls),
            "assistant_action_turns": assistant_action_count,
            "assistant_text_turns_omitted": assistant_text_turns_omitted,
            "tool_response_turns": tool_response_count,
            "parent_record_id": parent_record_id,
            "split": split,
            "source_turn_index": actions[0][0],
            "toolace_projection": TOOLACE_MULTITURN_ADAPTER_VERSION,
            "schema_fallback_tools": list(fallback_tools),
            "slot_values": {},
            "slot_policy": "omitted; only parent-record and prompt disjointness are claimed",
            "derivation": "all_canonical_assistant_actions_with_tool_history_v1",
            "quality": {
                "source_conversation_turns": len(raw["conversations"]),
                "tool_response_omitted": False,
                "assistant_text_omitted": not include_assistant_text,
            },
            "provenance": {
                "dataset": TOOLACE_DATASET,
                "subset": "source_record_disjoint_train" if split == "train" else "source_record_disjoint_eval",
                "revision": TOOLACE_REVISION,
                "record_id": parent_record_id,
                "url": TOOLACE_URL,
                "license": TOOLACE_LICENSE,
                "file_sha256": source_sha256,
                "source_line": record_index + 1,
            },
            "verified": False,
            "rule_verified": True,
            "model_verified": False,
            "environment_executed": False,
            "verification_scope": "tool_catalog_schema_and_multiturn_action_history_projection",
        },
    )
    assistant_training_turns(conversation)
    return conversation


def normalize_toolace_action_history_row(
    raw: object,
    *,
    record_index: int,
    split: str,
    source_sha256: str,
) -> Conversation | None:
    """Normalize action-relevant history while omitting non-action assistant prose."""

    conversation = normalize_toolace_multiturn_row(
        raw,
        record_index=record_index,
        split=split,
        source_sha256=source_sha256,
        include_assistant_text=False,
    )
    if conversation is not None:
        conversation.meta["toolace_projection"] = TOOLACE_ACTION_HISTORY_ADAPTER_VERSION
        conversation.meta["derivation"] = "canonical_assistant_actions_with_user_tool_history_v1"
        conversation.meta["verification_scope"] = (
            "tool_catalog_schema_and_action_history_projection"
        )
    return conversation


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _write_jsonl(path: Path, rows: list[Conversation]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                payload = (row.to_json() + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
                size += len(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest(), "rows": len(rows)}


def normalize_toolace_snapshot(
    input_path: str | Path,
    *,
    output_train: str | Path,
    output_eval: str | Path,
    manifest_path: str | Path,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    eval_modulo: int = 10,
    projection: str = "first_action",
) -> dict[str, Any]:
    """Normalize a byte-pinned ToolACE array into deterministic source-disjoint JSONL splits."""

    if eval_modulo < 2:
        raise ValueError("eval_modulo must be at least 2")
    if projection not in {"first_action", "multiturn", "action_history"}:
        raise ValueError("projection must be 'first_action', 'multiturn', or 'action_history'")
    adapter_version = (
        TOOLACE_ADAPTER_VERSION
        if projection == "first_action"
        else (
            TOOLACE_MULTITURN_ADAPTER_VERSION
            if projection == "multiturn"
            else TOOLACE_ACTION_HISTORY_ADAPTER_VERSION
        )
    )
    source = Path(input_path)
    identity = _identity(source)
    if expected_bytes is not None and identity["bytes"] != expected_bytes:
        raise ValueError("ToolACE source byte-size mismatch")
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError("ToolACE source SHA-256 mismatch")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("ToolACE source must be a top-level JSON array")
    train: list[Conversation] = []
    evaluation: list[Conversation] = []
    rejected = Counter()
    split_parent_ids: dict[str, set[str]] = {"train": set(), "eval": set()}
    split_prompts: dict[str, set[str]] = {"train": set(), "eval": set()}
    projection_stats = Counter()
    for index, row in enumerate(raw):
        action = _first_action(row)
        if action is None:
            rejected["no_strict_first_action"] += 1
            continue
        prompt = " ".join(action[0].casefold().split())
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).digest()
        split = "eval" if prompt_hash[0] % eval_modulo == 0 else "train"
        try:
            normalizer = (
                normalize_toolace_row
                if projection == "first_action"
                else (
                    normalize_toolace_multiturn_row
                    if projection == "multiturn"
                    else normalize_toolace_action_history_row
                )
            )
            conversation = normalizer(
                row, record_index=index, split=split, source_sha256=identity["sha256"]
            )
        except (TypeError, ValueError, SyntaxError, MemoryError, RecursionError):
            rejected["schema_or_projection_failure"] += 1
            continue
        if conversation is None:  # pragma: no cover - _first_action was already checked
            rejected["no_strict_first_action"] += 1
            continue
        target = evaluation if split == "eval" else train
        target.append(conversation)
        projection_stats["messages"] += len(conversation.messages)
        projection_stats["tool_response_messages"] += sum(
            message.role == Role.tool for message in conversation.messages
        )
        projection_stats["assistant_action_turns"] += sum(
            bool(message.tool_calls) for message in conversation.messages
        )
        projection_stats["assistant_text_turns_omitted"] += int(
            conversation.meta.get("assistant_text_turns_omitted", 0)
        )
        parent_id = str(conversation.meta["parent_record_id"])
        split_parent_ids[split].add(parent_id)
        split_prompts[split].add(prompt)
    if not train or not evaluation:
        raise ValueError("ToolACE projection must produce non-empty train and eval splits")
    if split_parent_ids["train"] & split_parent_ids["eval"]:
        raise AssertionError("ToolACE parent-record split overlap")
    if split_prompts["train"] & split_prompts["eval"]:
        raise AssertionError("ToolACE normalized prompt split overlap")
    def key(row: Conversation) -> str:
        return str(row.meta["parent_record_id"])

    train.sort(key=key)
    evaluation.sort(key=key)
    outputs = {
        "train": _write_jsonl(Path(output_train), train),
        "eval": _write_jsonl(Path(output_eval), evaluation),
    }
    manifest_core = {
        "kind": TOOLACE_MANIFEST_KIND,
        "schema_version": 1,
        "adapter_version": adapter_version,
        "projection_mode": projection,
        "dataset": TOOLACE_DATASET,
        "revision": TOOLACE_REVISION,
        "url": TOOLACE_URL,
        "license": TOOLACE_LICENSE,
        "license_url": TOOLACE_LICENSE_URL,
        "source": identity,
        "split_policy": "sha256(casefolded first-user prompt)[0] modulo 10; prompt and parent disjoint",
        "projection": (
            "first strict bracketed assistant action; tool response and later turns omitted"
            if projection == "first_action"
            else (
                "all strict bracketed assistant actions with user, assistant text, and tool-response history preserved"
                if projection == "multiturn"
                else "all strict bracketed assistant actions with user and tool-response history preserved; non-action assistant prose omitted"
            )
        ),
        "raw_rows": len(raw),
        "accepted_rows": len(train) + len(evaluation),
        "rejected_rows": sum(rejected.values()),
        "rejections": dict(sorted(rejected.items())),
        "outputs": outputs,
        "projection_stats": dict(sorted(projection_stats.items())),
        "split_audit": {
            "train_parent_records": len(split_parent_ids["train"]),
            "eval_parent_records": len(split_parent_ids["eval"]),
            "parent_record_overlap": 0,
            "train_prompts": len(split_prompts["train"]),
            "eval_prompts": len(split_prompts["eval"]),
            "prompt_overlap": 0,
            "slot_values_checked": False,
        },
        "claim_boundary": (
            "Source-record-disjoint ToolACE "
            + (
                "first-action"
                if projection == "first_action"
                else ("multi-turn" if projection == "multiturn" else "action-history")
            )
            + " projection for low-rate SFT/transfer diagnostics; not an official ToolACE split, "
            "BFCL score, multi-turn execution result, or native browser/mobile/desktop/MCP evaluation."
        ),
    }
    manifest, payload = self_hashed_manifest(manifest_core)
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{manifest_file.name}.", suffix=".tmp", dir=manifest_file.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_file)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest
