"""Deterministic, fail-closed prompt contracts for canonical conversations.

This module is intentionally in the data layer so training and evaluators can share one textual
materialization without importing one another.  The ``openai_full_catalog_v1`` contract renders
the complete OpenAI-style function catalog followed by role-preserving message history.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.model.tokenizer import (
    ASSISTANT,
    BPE_EOS,
    TOOL,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    USER,
)

LEGACY_CONVERSATION_PROMPT_CONTRACT = "legacy"
OPENAI_FULL_CATALOG_V1 = "openai_full_catalog_v1"
SUPPORTED_CONVERSATION_PROMPT_CONTRACTS = frozenset(
    {LEGACY_CONVERSATION_PROMPT_CONTRACT, OPENAI_FULL_CATALOG_V1}
)

SYSTEM = "<|system|>"
TOOL_CATALOG_OPEN = "<|tool_catalog|>"
TOOL_CATALOG_CLOSE = "</|tool_catalog|>"

RESERVED_PROMPT_MARKERS = (
    BPE_EOS,
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_CATALOG_OPEN,
    TOOL_CATALOG_CLOSE,
)

_SCHEMA_KEYS = frozenset(
    {
        "type",
        "description",
        "enum",
        "format",
        "properties",
        "required",
        "additionalProperties",
        "items",
    }
)
_SCALAR_TYPES = frozenset({"string", "integer", "number", "boolean", "null"})
_SCHEMA_TYPES = _SCALAR_TYPES | {"object", "array"}


def _assert_no_reserved_prompt_marker(value: str, *, label: str) -> None:
    """Reject text that could escape or counterfeit the full prompt's framing."""

    for marker in RESERVED_PROMPT_MARKERS:
        if marker in value:
            raise ValueError(f"{label} contains reserved prompt marker {marker!r}")


def _assert_no_reserved_json_marker(value: Any, *, label: str) -> None:
    """Walk JSON-like data and check every string key and value."""

    if isinstance(value, str):
        _assert_no_reserved_prompt_marker(value, label=label)
        return
    if isinstance(value, Mapping):
        for index, (key, child) in enumerate(value.items()):
            if isinstance(key, str):
                _assert_no_reserved_prompt_marker(key, label=f"{label} key {index}")
            _assert_no_reserved_json_marker(child, label=f"{label} value {index}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_reserved_json_marker(child, label=f"{label} item {index}")


@dataclass(frozen=True)
class AssistantTrainingExample:
    """One exact decode prompt and its next assistant body."""

    message_index: int
    prompt: str
    body: str


@dataclass(frozen=True)
class AssistantTrainingTurn:
    """One assistant target with the catalog-independent suffix of its decode prompt."""

    message_index: int
    prompt_suffix: str
    body: str


@dataclass(frozen=True)
class _FunctionCatalogCacheEntry:
    tools: Sequence[ToolSpec]
    registry: dict[str, ToolSpec]
    text: str


class FunctionCatalogCache:
    """Reuse validation/rendering for interned, verified immutable tool catalogs.

    Verified conversation artifacts intern their recursively read-only catalog list, so a
    50,000-row artifact normally has one catalog object. Mutable caller-owned lists deliberately
    bypass this identity cache: changing a schema between calls must never reuse stale validation.
    """

    def __init__(self) -> None:
        self._entries: dict[int, _FunctionCatalogCacheEntry] = {}

    def entry(self, tools: Sequence[ToolSpec]) -> _FunctionCatalogCacheEntry:
        cacheable = getattr(tools, "_localagent_verified_read_only", False) is True
        key = id(tools)
        if cacheable:
            cached = self._entries.get(key)
            if cached is not None and cached.tools is tools:
                return cached
        registry = validate_tool_catalog(tools)
        entry = _FunctionCatalogCacheEntry(
            tools=tools,
            registry=registry,
            text=_render_validated_function_catalog(tools),
        )
        if cacheable:
            self._entries[key] = entry
        return entry

    @property
    def unique_catalogs(self) -> int:
        return len(self._entries)


def resolve_conversation_prompt_contract(value: Any = None) -> str:
    """Normalize a configured contract, preserving legacy behavior when the key is absent."""

    if value is None:
        return LEGACY_CONVERSATION_PROMPT_CONTRACT
    if not isinstance(value, str):
        raise TypeError("data.conversation_prompt_contract must be text")
    if value not in SUPPORTED_CONVERSATION_PROMPT_CONTRACTS:
        raise ValueError(
            "data.conversation_prompt_contract must be one of "
            f"{sorted(SUPPORTED_CONVERSATION_PROMPT_CONTRACTS)}, got {value!r}"
        )
    return value


def assert_prompt_contract_tokenizer(tokenizer, prompt_contract: str | None) -> str:
    """Validate tokenizer-level invariants and return the normalized contract."""

    contract = resolve_conversation_prompt_contract(prompt_contract)
    if contract == OPENAI_FULL_CATALOG_V1:
        marker_ids = tokenizer.encode(BPE_EOS)
        if marker_ids != [tokenizer.eos_id]:
            raise ValueError(
                "openai_full_catalog_v1 requires a BPE tokenizer whose canonical "
                f"{BPE_EOS!r} marker encodes to exactly eos_id"
            )
        for suffix in (
            SYSTEM + "boundary-check",
            USER + "boundary-check",
            TOOL + "boundary-check",
            ASSISTANT + "boundary-check",
        ):
            if tokenizer.encode(BPE_EOS + suffix) != [tokenizer.eos_id, *tokenizer.encode(suffix)]:
                raise ValueError(
                    "openai_full_catalog_v1 requires an atomic EOS prompt boundary "
                    f"before suffix {suffix!r}"
                )
    return contract


def _json_kind(value: Any) -> str | None:
    """Return the JSON data-model kind, rejecting Python-only and non-finite values."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array" if all(_json_kind(item) is not None for item in value) else None
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            return None
        return "object" if all(_json_kind(item) is not None for item in value.values()) else None
    return None


def _finite_json(value: Any) -> bool:
    return _json_kind(value) is not None


def _json_semantic_equal(left: Any, right: Any) -> bool:
    """Compare recursively using JSON types, where booleans are never numbers."""

    left_kind = _json_kind(left)
    right_kind = _json_kind(right)
    if left_kind is None or left_kind != right_kind:
        return False
    if left_kind == "array":
        return len(left) == len(right) and all(
            _json_semantic_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if left_kind == "object":
        return left.keys() == right.keys() and all(
            _json_semantic_equal(left[key], right[key]) for key in left
        )
    return bool(left == right)


def validate_json_schema(schema: Any, *, label: str) -> dict[str, Any]:
    """Validate the recursively supported JSON Schema subset or fail closed."""

    if not isinstance(schema, Mapping):
        raise TypeError(f"{label} must be a JSON Schema object")
    if any(not isinstance(key, str) for key in schema):
        raise ValueError(f"{label} contains a non-string keyword")
    unsupported = sorted(set(schema) - _SCHEMA_KEYS)
    if unsupported:
        raise ValueError(f"{label} contains unsupported JSON Schema keywords: {unsupported}")
    normalized = dict(schema)
    schema_type = normalized.get("type")
    if schema_type is not None and (
        not isinstance(schema_type, str) or schema_type not in _SCHEMA_TYPES
    ):
        raise ValueError(f"{label}.type is unsupported: {schema_type!r}")

    if "enum" in normalized:
        enum = normalized["enum"]
        if not isinstance(enum, list) or not enum:
            raise ValueError(f"{label}.enum must be a non-empty JSON array")
        if not _finite_json(enum):
            raise ValueError(f"{label}.enum must contain only finite JSON values")

    if "description" in normalized and not isinstance(normalized["description"], str):
        raise ValueError(f"{label}.description must be text")

    if "format" in normalized and (
        schema_type != "string" or not isinstance(normalized["format"], str)
    ):
        raise ValueError(f"{label}.format is supported only as a string annotation")

    object_keywords = {"properties", "required", "additionalProperties"} & set(normalized)
    if object_keywords and schema_type != "object":
        raise ValueError(f"{label} uses object keywords without type='object'")
    if "items" in normalized and schema_type != "array":
        raise ValueError(f"{label} uses items without type='array'")

    if schema_type == "object":
        properties = normalized.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"{label}.properties must be an object")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{label}.properties contains an invalid property name")
            validate_json_schema(child, label=f"{label}.properties[{name!r}]")
        required = normalized.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) or not name for name in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError(f"{label}.required must contain unique non-empty strings")
        unknown_required = sorted(set(required) - set(properties))
        if unknown_required:
            raise ValueError(f"{label}.required names undefined properties: {unknown_required}")
        additional = normalized.get("additionalProperties", True)
        if not isinstance(additional, (bool, Mapping)):
            raise ValueError(f"{label}.additionalProperties must be boolean or a supported schema")
        if isinstance(additional, Mapping):
            validate_json_schema(
                additional,
                label=f"{label}.additionalProperties",
            )
    elif schema_type == "array":
        if "items" not in normalized:
            raise ValueError(f"{label} array schema must declare items")
        validate_json_schema(normalized["items"], label=f"{label}.items")

    return normalized


def _enum_matches(value: Any, candidates: Sequence[Any]) -> bool:
    return any(_json_semantic_equal(value, candidate) for candidate in candidates)


def schema_matches(value: Any, schema: Mapping[str, Any]) -> bool:
    """Return whether a finite JSON value satisfies an already validated schema."""

    if not _finite_json(value):
        return False
    if "enum" in schema and not _enum_matches(value, schema["enum"]):
        return False
    schema_type = schema.get("type")
    if schema_type is None:
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return not isinstance(value, float) or math.isfinite(value)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    if schema_type == "array":
        return isinstance(value, list) and all(
            schema_matches(item, schema["items"]) for item in value
        )
    if schema_type != "object" or not isinstance(value, dict):
        return False

    properties = schema.get("properties", {})
    if any(name not in value for name in schema.get("required", [])):
        return False
    for name, item in value.items():
        child = properties.get(name)
        if child is not None:
            if not schema_matches(item, child):
                return False
            continue
        additional = schema.get("additionalProperties", True)
        if additional is False:
            return False
        if isinstance(additional, Mapping) and not schema_matches(item, additional):
            return False
    return True


def validate_tool_catalog(
    tools: Sequence[ToolSpec],
    *,
    label: str = "function catalog",
) -> dict[str, ToolSpec]:
    """Validate a complete function catalog and return its name registry."""

    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    registry: dict[str, ToolSpec] = {}
    for tool in tools:
        if not isinstance(tool, ToolSpec):
            raise TypeError(f"{label} entries must be ToolSpec values")
        if not isinstance(tool.name, str) or not tool.name:
            raise ValueError(f"{label} contains a tool with an invalid name")
        _assert_no_reserved_prompt_marker(tool.name, label=f"{label} tool name")
        if tool.name in registry:
            raise ValueError(f"{label} contains duplicate tool name {tool.name!r}")
        if not isinstance(tool.description, str):
            raise TypeError(f"{label} tool {tool.name!r} description must be text")
        _assert_no_reserved_prompt_marker(
            tool.description,
            label=f"{label} tool {tool.name!r} description",
        )
        _assert_no_reserved_json_marker(
            tool.parameters,
            label=f"{label} tool {tool.name!r} parameters",
        )
        parameters = validate_json_schema(
            tool.parameters,
            label=f"{label} tool {tool.name!r} parameters",
        )
        if parameters.get("type") != "object":
            raise ValueError(f"{label} tool {tool.name!r} parameters must have type='object'")
        registry[tool.name] = tool
    return registry


def _render_validated_function_catalog(tools: Sequence[ToolSpec]) -> str:
    catalog = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
    }
    encoded = json.dumps(
        catalog,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return TOOL_CATALOG_OPEN + encoded + TOOL_CATALOG_CLOSE


def render_function_catalog(tools: Sequence[ToolSpec]) -> str:
    """Render every tool in deterministic OpenAI-style function-catalog JSON."""

    validate_tool_catalog(tools)
    return _render_validated_function_catalog(tools)


def render_tool_calls(calls: Sequence[ToolCall]) -> str:
    """Render ordered tool calls using the canonical LocalAgent envelope."""

    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise TypeError("assistant tool_calls must be a sequence")
    parts = []
    for call in calls:
        if not isinstance(call, ToolCall):
            raise TypeError("assistant tool_calls entries must be ToolCall values")
        if not isinstance(call.name, str) or not call.name:
            raise ValueError("assistant tool call name must be non-empty text")
        _assert_no_reserved_prompt_marker(call.name, label="assistant tool call name")
        if not isinstance(call.arguments, dict):
            raise TypeError("assistant tool call arguments must be an object")
        _assert_no_reserved_json_marker(
            call.arguments,
            label="assistant tool call arguments",
        )
        try:
            encoded = json.dumps(
                {"name": call.name, "arguments": call.arguments},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "assistant tool call arguments must contain finite JSON values"
            ) from error
        parts.append(TOOL_CALL_OPEN + encoded + TOOL_CALL_CLOSE)
    return "".join(parts)


def _validate_message(
    message: Message,
    *,
    index: int,
    registry: Mapping[str, ToolSpec],
) -> None:
    if not isinstance(message, Message):
        raise TypeError(f"message {index} must be a Message")
    if not isinstance(message.content, str):
        raise TypeError(f"message {index} content must be text")
    _assert_no_reserved_prompt_marker(message.content, label=f"message {index} content")
    if message.role == Role.tool:
        if message.tool_response is not None and not isinstance(message.tool_response, str):
            raise ValueError(f"message {index} tool_response must be text or null")
        if message.tool_response is not None:
            _assert_no_reserved_prompt_marker(
                message.tool_response,
                label=f"message {index} tool_response",
            )
    elif message.role == Role.assistant:
        if message.tool_calls and message.content:
            raise ValueError(f"message {index} mixes tool calls and assistant text")
        render_tool_calls(message.tool_calls)
        for call in message.tool_calls:
            tool = registry.get(call.name)
            if tool is None:
                raise ValueError(f"message {index} references unknown tool {call.name!r}")
            if not schema_matches(call.arguments, tool.parameters):
                raise ValueError(f"message {index} arguments violate the schema for {call.name!r}")
    elif message.role not in {Role.system, Role.user}:
        raise ValueError(f"message {index} has unsupported role {message.role!r}")


def render_message_history(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
) -> str:
    """Render exact role-preserving history without a trailing assistant marker."""

    registry = validate_tool_catalog(tools)
    parts = []
    for index, message in enumerate(messages):
        _validate_message(message, index=index, registry=registry)
        if message.role == Role.system:
            parts.append(SYSTEM + message.content)
        elif message.role == Role.user:
            parts.append(USER + message.content)
        elif message.role == Role.tool:
            parts.append(
                TOOL + TOOL_RESPONSE_OPEN + (message.tool_response or "") + TOOL_RESPONSE_CLOSE
            )
        else:
            body = render_tool_calls(message.tool_calls) if message.tool_calls else message.content
            parts.append(ASSISTANT + body + BPE_EOS)
    return "".join(parts)


def render_agent_decode_prompt(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
) -> str:
    """Render a complete catalog, exact message prefix, and next-assistant marker."""

    return (
        render_function_catalog(tools)
        + BPE_EOS
        + render_message_history(messages, tools)
        + ASSISTANT
    )


def assistant_training_examples(
    conversation: Conversation,
    *,
    catalog_cache: FunctionCatalogCache | None = None,
) -> tuple[AssistantTrainingExample, ...]:
    """Materialize one eval-parity prompt/target pair per assistant decision."""

    cache = catalog_cache if catalog_cache is not None else FunctionCatalogCache()
    entry = cache.entry(conversation.tools)
    catalog = entry.text + BPE_EOS
    return tuple(
        AssistantTrainingExample(
            message_index=turn.message_index,
            prompt=catalog + turn.prompt_suffix,
            body=turn.body,
        )
        for turn in _assistant_training_turns(conversation, entry.registry)
    )


def assistant_training_turns(
    conversation: Conversation,
    *,
    catalog_cache: FunctionCatalogCache | None = None,
) -> tuple[AssistantTrainingTurn, ...]:
    """Materialize assistant targets without repeating the catalog text in every prompt."""

    if not isinstance(conversation, Conversation):
        raise TypeError("conversation must be a Conversation")
    if catalog_cache is None:
        registry = validate_tool_catalog(
            conversation.tools,
            label="conversation function catalog",
        )
    else:
        registry = catalog_cache.entry(conversation.tools).registry
    return _assistant_training_turns(conversation, registry)


def _assistant_training_turns(
    conversation: Conversation,
    registry: Mapping[str, ToolSpec],
) -> tuple[AssistantTrainingTurn, ...]:
    for index, message in enumerate(conversation.messages):
        _validate_message(message, index=index, registry=registry)

    examples = []
    history_parts: list[str] = []
    for index, message in enumerate(conversation.messages):
        if message.role == Role.assistant:
            body = render_tool_calls(message.tool_calls) if message.tool_calls else message.content
            examples.append(
                AssistantTrainingTurn(
                    message_index=index,
                    prompt_suffix="".join(history_parts) + ASSISTANT,
                    body=body,
                )
            )
            history_parts.append(ASSISTANT + body + BPE_EOS)
        elif message.role == Role.system:
            history_parts.append(SYSTEM + message.content)
        elif message.role == Role.user:
            history_parts.append(USER + message.content)
        elif message.role == Role.tool:
            history_parts.append(
                TOOL + TOOL_RESPONSE_OPEN + (message.tool_response or "") + TOOL_RESPONSE_CLOSE
            )
        else:  # pragma: no cover - _validate_message rejects unsupported roles
            raise AssertionError(f"unhandled message role {message.role!r}")
    return tuple(examples)
