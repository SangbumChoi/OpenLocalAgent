"""Pinned, offline ingestion for public tool-use and browser-action datasets.

The network is deliberately outside this module.  Acquisition is a separate, auditable step:
each local source snapshot must be bound by byte count, SHA-256, dataset revision, URL, and
license.  This module normalizes those immutable bytes into the project's one
``Conversation`` interchange format.

Three adapters are supported:

``xlam_v1``
    Salesforce/xlam-function-calling-60k TRAIN records.  The source's stringified query, tools,
    and answers become a user turn and one canonical assistant tool-call turn.

``mind2web_v1``
    Mind2Web TRAIN tasks.  Grounded CLICK/TYPE/SELECT steps become realistic multi-step browser
    trajectories.  Tasks with an action that lacks a positive target are rejected instead of
    inventing a label.

``localagent_v1``
    A small, explicit JSONL interchange used by audited adapters and offline tests.  It supports
    action traces plus abstention/irrelevance negatives and declares split-sensitive slot values.

``agentnet_v1``
    Evaluation-only normalization for the public OpenCUA AgentNet desktop trajectories.  It
    preserves low-level coordinate actions as ``agentnet_*`` tools and requires a textual
    observation; it does not claim screenshot-to-WebGPU grounding.

Known benchmark/test sources are never accepted as training input.  Exact prompt-denylist
artifacts can additionally be pinned in the build config, so a public TRAIN snapshot cannot
silently absorb a frozen evaluation prompt.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO

import yaml

from localagent.data.agentnet import AGENTNET_REVISION, normalize_agentnet_record
from localagent.data.conversation_artifact import (
    FileIdentity,
    canonical_json_bytes,
    conversation_semantic_sha256,
    self_hashed_manifest,
)
from localagent.data.pretrain_corpus import read_evaluation_denylist
from localagent.data.prompt_contract import assistant_training_turns
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec

PUBLIC_AGENT_ARTIFACT_KIND = "localagent_public_agent_mixture"
PUBLIC_AGENT_SCHEMA_VERSION = 1
PUBLIC_AGENT_GENERATOR = "public_agent_snapshot_v1"
VERIFICATION_SCOPE = "schema_catalog_arguments_sequence_and_split_slots"

XLAM_DATASET = "Salesforce/xlam-function-calling-60k"
XLAM_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
MIND2WEB_DATASET = "osunlp/Mind2Web"
MIND2WEB_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
AGENTNET_DATASET = "xlangai/AgentNet"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_JSONL_ROW_BYTES = 16 * 1024 * 1024
_ADAPTERS = frozenset({"xlam_v1", "mind2web_v1", "localagent_v1", "agentnet_v1"})
_BEHAVIORS = frozenset({"action", "abstention", "irrelevance"})
_SPLITS = frozenset({"train", "eval"})

_TRAIN_WRAPPERS = (
    "Please handle this as a real work request. {request}",
    "Take care of this operational task and report the result. {request}",
    "Work through this request carefully. {request}",
    "Complete this task using only the relevant tools. {request}",
)
_TRAIN_CONTEXT_WRAPPERS = (
    "Context: you are assisting with {domain}. Request: {request}",
    "This is a {domain} workflow. Please complete it: {request}",
    "In the {domain} queue, the next request is: {request}",
    "Treat this as a live {domain} case: {request}",
)


@dataclass(frozen=True)
class PublicSourceSnapshot:
    """One immutable local snapshot and the public provenance that authorizes its use."""

    source_id: str
    dataset: str
    subset: str
    revision: str
    url: str
    license: str
    license_url: str
    adapter: str
    split: Literal["train", "eval"]
    path: Path
    declared_path: str
    bytes: int
    sha256: str
    max_records: int | None = None
    max_actions_per_record: int | None = None

    def provenance(self, *, record_id: str, source_line: int) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "subset": self.subset,
            "revision": self.revision,
            "record_id": record_id,
            "url": self.url,
            "license": self.license,
            "file_sha256": self.sha256,
            "source_line": source_line,
        }


@dataclass(frozen=True)
class PublicAgentBuildResult:
    """Published split files and their self-hashed audit manifest."""

    conversations: Mapping[str, tuple[Conversation, ...]]
    outputs: Mapping[str, Path]
    manifest_path: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _NormalizedRecord:
    record_id: str
    domain: str
    behavior: str
    capabilities: tuple[str, ...]
    slot_values: Mapping[str, tuple[Any, ...]]
    tools: tuple[ToolSpec, ...]
    messages: tuple[Message, ...]
    quality: Mapping[str, Any]
    source_line: int


def _strict_json_loads(value: str, *, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {constant}")

    try:
        return json.loads(
            value,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error


def _canonical_json_value(value: Any, *, label: str) -> Any:
    """Return recursively key-sorted, finite JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _canonical_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{label} object keys must be strings")
        return {
            key: _canonical_json_value(value[key], label=f"{label}.{key}")
            for key in sorted(value)
        }
    raise TypeError(f"{label} contains non-JSON value {type(value).__name__}")


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _stable_index(*parts: object, modulo: int) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def _file_path(base: Path, value: object, *, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return value, path.resolve()


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    result = _non_negative_int(value, label=label)
    if result < 1:
        raise ValueError(f"{label} must be positive when provided")
    return result


def _source_from_config(raw: object, *, index: int, base: Path) -> PublicSourceSnapshot:
    if not isinstance(raw, dict):
        raise TypeError(f"sources[{index}] must be a mapping")
    allowed = {
        "source_id",
        "dataset",
        "subset",
        "revision",
        "url",
        "license",
        "license_url",
        "adapter",
        "split",
        "path",
        "bytes",
        "sha256",
        "max_records",
        "max_actions_per_record",
    }
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ValueError(f"sources[{index}] has unknown keys: {extra}")
    required = allowed - {"max_records", "max_actions_per_record"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"sources[{index}] is missing keys: {missing}")
    strings: dict[str, str] = {}
    for key in (
        "source_id",
        "dataset",
        "subset",
        "revision",
        "url",
        "license",
        "license_url",
        "adapter",
        "split",
        "sha256",
    ):
        value = raw[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"sources[{index}].{key} must be non-empty text")
        strings[key] = value
    if strings["adapter"] not in _ADAPTERS:
        raise ValueError(f"sources[{index}].adapter is unsupported: {strings['adapter']!r}")
    if strings["split"] not in _SPLITS:
        raise ValueError(f"sources[{index}].split must be train or eval")
    if not strings["url"].startswith("https://"):
        raise ValueError(f"sources[{index}].url must use HTTPS")
    if not strings["license_url"].startswith("https://"):
        raise ValueError(f"sources[{index}].license_url must use HTTPS")
    if _SHA256_RE.fullmatch(strings["sha256"]) is None:
        raise ValueError(f"sources[{index}].sha256 must be a lowercase SHA-256")
    byte_count = _non_negative_int(raw["bytes"], label=f"sources[{index}].bytes")
    declared_path, path = _file_path(base, raw["path"], label=f"sources[{index}].path")
    source = PublicSourceSnapshot(
        source_id=strings["source_id"],
        dataset=strings["dataset"],
        subset=strings["subset"],
        revision=strings["revision"],
        url=strings["url"],
        license=strings["license"],
        license_url=strings["license_url"],
        adapter=strings["adapter"],
        split=strings["split"],  # type: ignore[arg-type]
        path=path,
        declared_path=declared_path,
        bytes=byte_count,
        sha256=strings["sha256"],
        max_records=_optional_positive_int(
            raw.get("max_records"),
            label=f"sources[{index}].max_records",
        ),
        max_actions_per_record=_optional_positive_int(
            raw.get("max_actions_per_record"),
            label=f"sources[{index}].max_actions_per_record",
        ),
    )
    _enforce_known_source_policy(source)
    return source


def _enforce_known_source_policy(source: PublicSourceSnapshot) -> None:
    """Fail closed around supported public TRAIN sources and known eval-only families."""

    dataset_key = source.dataset.casefold()
    if dataset_key == XLAM_DATASET.casefold():
        if (
            source.adapter != "xlam_v1"
            or source.split != "train"
            or source.revision != XLAM_REVISION
            or source.license.casefold() != "cc-by-4.0"
        ):
            raise ValueError(
                f"{XLAM_DATASET} must use pinned TRAIN/xlam_v1/CC-BY-4.0 policy"
            )
    if dataset_key == MIND2WEB_DATASET.casefold():
        normalized_path = source.declared_path.replace("\\", "/")
        if (
            source.adapter != "mind2web_v1"
            or source.split != "train"
            or source.revision != MIND2WEB_REVISION
            or source.license.casefold() != "cc-by-4.0"
            or "/train_" not in normalized_path
        ):
            raise ValueError(
                f"{MIND2WEB_DATASET} must use pinned data/train/train_*.json only"
            )
    if dataset_key == AGENTNET_DATASET.casefold():
        if (
            source.adapter != "agentnet_v1"
            or source.split != "eval"
            or source.revision != AGENTNET_REVISION
            or source.license.casefold() != "mit"
        ):
            raise ValueError(
                f"{AGENTNET_DATASET} is evaluation-only and must use the pinned AgentNet revision"
            )
    if "weblinx" in dataset_key and source.split == "train":
        raise ValueError("WebLINX is eval-only/non-default because its data license is noncommercial")
    if "bfcl" in dataset_key or "berkeley-function-calling" in dataset_key:
        if source.split == "train":
            raise ValueError("BFCL benchmark material must never be used for training")


@contextmanager
def _verified_text_source(
    source: PublicSourceSnapshot,
    *,
    max_source_bytes: int,
) -> Iterator[TextIO]:
    if source.bytes > max_source_bytes:
        raise ValueError(
            f"source {source.source_id!r} exceeds max_source_bytes={max_source_bytes}"
        )
    try:
        raw: BinaryIO = source.path.open("rb")
    except OSError as error:
        raise ValueError(f"source snapshot is unavailable: {source.path}") from error
    text: io.TextIOWrapper | None = None
    try:
        observed = os.fstat(raw.fileno())
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"source snapshot is not a regular file: {source.path}")
        if observed.st_size != source.bytes:
            raise ValueError(f"source {source.source_id!r} byte-size mismatch")
        digest = hashlib.sha256()
        observed_bytes = 0
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            observed_bytes += len(chunk)
            if observed_bytes > max_source_bytes:
                raise ValueError(
                    f"source {source.source_id!r} exceeds max_source_bytes={max_source_bytes}"
                )
            digest.update(chunk)
        if observed_bytes != source.bytes or digest.hexdigest() != source.sha256:
            raise ValueError(f"source {source.source_id!r} SHA-256 identity mismatch")
        raw.seek(0)
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", newline="")
        yield text
    except UnicodeDecodeError as error:
        raise ValueError(f"source {source.source_id!r} is not valid UTF-8") from error
    finally:
        if text is not None:
            text.close()
        else:
            raw.close()


def _iter_json_array(handle: TextIO, *, label: str) -> Iterator[tuple[int, Any]]:
    """Incrementally parse a top-level JSON array using only the standard library."""

    decoder = json.JSONDecoder(
        object_pairs_hook=lambda pairs: _unique_pairs(pairs, label=label),
        parse_constant=lambda value: _reject_constant(value, label=label),
    )
    buffer = ""
    position = 0
    row_number = 0
    state = "open"
    eof = False

    def refill() -> bool:
        nonlocal buffer, position, eof
        if eof:
            return False
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = handle.read(1024 * 1024)
        if chunk:
            buffer += chunk
            return True
        eof = True
        return False

    while True:
        while position >= len(buffer) and refill():
            pass
        while position < len(buffer) and buffer[position].isspace():
            position += 1
            if position >= len(buffer):
                refill()
        if state == "open":
            if position >= len(buffer) or buffer[position] != "[":
                raise ValueError(f"{label} must be a top-level JSON array")
            position += 1
            state = "value_or_end"
            continue
        while position >= len(buffer) and refill():
            pass
        while position < len(buffer) and buffer[position].isspace():
            position += 1
            if position >= len(buffer):
                refill()
        if state == "value_or_end" and position < len(buffer) and buffer[position] == "]":
            position += 1
            state = "done"
            continue
        if state == "comma_or_end":
            if position < len(buffer) and buffer[position] == ",":
                position += 1
                state = "value_or_end"
                continue
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                state = "done"
                continue
            raise ValueError(f"{label} expected ',' or ']' after array item {row_number}")
        if state == "done":
            trailing = buffer[position:] + handle.read()
            if trailing.strip():
                raise ValueError(f"{label} has data after its top-level array")
            return
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError as error:
                if not refill():
                    raise ValueError(
                        f"{label} has invalid JSON at array item {row_number + 1}: {error}"
                    ) from error
        position = end
        row_number += 1
        yield row_number, value
        state = "comma_or_end"


def _iter_jsonl(handle: TextIO, *, label: str) -> Iterator[tuple[int, Any]]:
    """Yield strict JSON objects from a UTF-8 JSONL source."""

    for source_line, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > _MAX_JSONL_ROW_BYTES:
            raise ValueError(f"{label} line {source_line} exceeds row byte cap")
        yield source_line, _strict_json_loads(
            line,
            label=f"{label} line {source_line}",
        )


def _unique_pairs(pairs: list[tuple[str, Any]], *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str, *, label: str) -> None:
    raise ValueError(f"{label} contains non-finite JSON constant {value}")


def _decode_nested_json(value: Any, *, label: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in '[{"':
        return value
    return _strict_json_loads(stripped, label=label)


def _json_type_schema(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    normalized = value.casefold().replace(" ", "")
    direct = {
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
    }
    if normalized in direct:
        result: dict[str, Any] = {"type": direct[normalized]}
        if result["type"] == "array":
            result["items"] = {}
        return result
    match = re.fullmatch(r"(?:list|array)\[(.+)]", normalized)
    if match:
        return {"type": "array", "items": _json_type_schema(match.group(1), label=label)}
    # Unknown APIGen annotations remain unconstrained instead of being guessed.
    return {}


def _xlam_tool(raw: object, *, label: str) -> ToolSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be an object")
    name = raw.get("name")
    description = raw.get("description", "")
    parameters = raw.get("parameters", {})
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.name must be non-empty text")
    if not isinstance(description, str):
        raise TypeError(f"{label}.description must be text")
    if not isinstance(parameters, dict):
        raise TypeError(f"{label}.parameters must be an object")
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), dict):
        schema = _canonical_json_value(parameters, label=f"{label}.parameters")
    else:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter_name in sorted(parameters):
            parameter = parameters[parameter_name]
            if not isinstance(parameter_name, str) or not isinstance(parameter, dict):
                raise TypeError(f"{label}.parameters entries must be named objects")
            property_schema = _json_type_schema(
                parameter.get("type"),
                label=f"{label}.parameters.{parameter_name}.type",
            )
            parameter_description = parameter.get("description")
            if isinstance(parameter_description, str):
                property_schema["description"] = parameter_description
            enum = parameter.get("enum")
            if isinstance(enum, list):
                property_schema["enum"] = _canonical_json_value(
                    enum,
                    label=f"{label}.parameters.{parameter_name}.enum",
                )
            properties[parameter_name] = property_schema
            if parameter.get("required") is True:
                required.append(parameter_name)
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    return ToolSpec(name=name, description=description, parameters=schema)


def _leaf_slots(value: Any, *, prefix: str, out: dict[str, list[Any]]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            _leaf_slots(value[key], prefix=child, out=out)
    elif isinstance(value, list):
        for item in value:
            _leaf_slots(item, prefix=prefix, out=out)
    elif value is not None:
        out.setdefault(prefix or "value", []).append(_canonical_json_value(value, label=prefix))


def _xlam_record(raw: object, *, source_line: int) -> _NormalizedRecord:
    if not isinstance(raw, dict):
        raise TypeError(f"xLAM record {source_line} must be an object")
    query = _decode_nested_json(raw.get("query"), label=f"xLAM record {source_line}.query")
    tools_raw = _decode_nested_json(raw.get("tools"), label=f"xLAM record {source_line}.tools")
    answers = _decode_nested_json(
        raw.get("answers"),
        label=f"xLAM record {source_line}.answers",
    )
    if not isinstance(query, str) or not query:
        raise ValueError(f"xLAM record {source_line}.query must decode to text")
    if not isinstance(tools_raw, list) or not tools_raw:
        raise ValueError(f"xLAM record {source_line}.tools must decode to a non-empty list")
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"xLAM record {source_line}.answers must decode to a non-empty list")
    tools = tuple(
        sorted(
            (
                _xlam_tool(tool, label=f"xLAM record {source_line}.tools[{index}]")
                for index, tool in enumerate(tools_raw)
            ),
            key=lambda tool: tool.name,
        )
    )
    calls: list[ToolCall] = []
    slot_values: dict[str, list[Any]] = {}
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise TypeError(f"xLAM record {source_line}.answers[{index}] must be an object")
        name = answer.get("name")
        arguments = _decode_nested_json(
            answer.get("arguments", {}),
            label=f"xLAM record {source_line}.answers[{index}].arguments",
        )
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError(f"xLAM record {source_line}.answers[{index}] is malformed")
        arguments = _canonical_json_value(
            arguments,
            label=f"xLAM record {source_line}.answers[{index}].arguments",
        )
        calls.append(ToolCall(name=name, arguments=arguments))
        _leaf_slots(arguments, prefix=name, out=slot_values)
    record_id = str(raw.get("id", source_line - 1))
    prefixes = {call.name.split(".", 1)[0] for call in calls}
    domain = next(iter(prefixes)) if len(prefixes) == 1 else "function_calling"
    return _NormalizedRecord(
        record_id=record_id,
        domain=domain,
        behavior="action",
        capabilities=tuple(sorted({call.name for call in calls})),
        slot_values={
            key: tuple(values) for key, values in sorted(slot_values.items())
        },
        tools=tools,
        messages=(
            Message(role=Role.user, content=query),
            Message(role=Role.assistant, tool_calls=calls),
        ),
        quality={"source_verified": True},
        source_line=source_line,
    )


def _mind2web_tools() -> tuple[ToolSpec, ...]:
    target = {
        "target_id": {
            "type": "string",
            "description": "Backend node id of the grounded element.",
        }
    }
    return (
        ToolSpec(
            name="web_click",
            description="Click a grounded web-page element.",
            parameters={
                "type": "object",
                "properties": target,
                "required": ["target_id"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="web_select",
            description="Select an option on a grounded web-page element.",
            parameters={
                "type": "object",
                "properties": {
                    **target,
                    "value": {"type": "string", "description": "Option value to select."},
                },
                "required": ["target_id", "value"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="web_type",
            description="Type text into a grounded web-page element.",
            parameters={
                "type": "object",
                "properties": {
                    **target,
                    "text": {"type": "string", "description": "Text to enter."},
                },
                "required": ["target_id", "text"],
                "additionalProperties": False,
            },
        ),
    )


def _mind2web_target(action: Mapping[str, Any], *, label: str) -> str:
    candidates = action.get("pos_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{label} has no positive grounded target")
    valid = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("backend_node_id"), str)
        and candidate["backend_node_id"]
    ]
    if not valid:
        raise ValueError(f"{label} has no positive candidate backend_node_id")
    ordered = sorted(
        enumerate(valid),
        key=lambda pair: (
            not bool(pair[1].get("is_original_target")),
            not bool(pair[1].get("is_top_level_target")),
            pair[0],
        ),
    )
    return str(ordered[0][1]["backend_node_id"])


def _mind2web_record(
    raw: object,
    *,
    source_line: int,
    max_actions: int | None,
) -> _NormalizedRecord:
    if not isinstance(raw, dict):
        raise TypeError(f"Mind2Web record {source_line} must be an object")
    record_id = raw.get("annotation_id")
    task = raw.get("confirmed_task")
    actions = raw.get("actions")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"Mind2Web record {source_line}.annotation_id is invalid")
    if not isinstance(task, str) or not task:
        raise ValueError(f"Mind2Web record {source_line}.confirmed_task is invalid")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"Mind2Web record {source_line}.actions must be non-empty")
    if max_actions is not None:
        actions = actions[:max_actions]
    action_reprs = raw.get("action_reprs", [])
    messages: list[Message] = [Message(role=Role.user, content=task)]
    capabilities: set[str] = set()
    slot_values: dict[str, list[Any]] = {}
    for index, action in enumerate(actions):
        label = f"Mind2Web record {source_line}.actions[{index}]"
        if not isinstance(action, dict):
            raise TypeError(f"{label} must be an object")
        operation = action.get("operation")
        if not isinstance(operation, dict):
            raise TypeError(f"{label}.operation must be an object")
        op = operation.get("op")
        value = operation.get("value", "")
        if not isinstance(op, str) or not isinstance(value, str):
            raise ValueError(f"{label}.operation is malformed")
        target_id = _mind2web_target(action, label=label)
        normalized_op = op.upper()
        if normalized_op == "CLICK":
            name = "web_click"
            arguments = {"target_id": target_id}
        elif normalized_op == "TYPE":
            name = "web_type"
            arguments = {"target_id": target_id, "text": value}
        elif normalized_op == "SELECT":
            name = "web_select"
            arguments = {"target_id": target_id, "value": value}
        else:
            raise ValueError(f"{label}.operation.op is unsupported: {op!r}")
        messages.append(Message(role=Role.assistant, tool_calls=[ToolCall(name, arguments)]))
        action_repr = (
            action_reprs[index]
            if isinstance(action_reprs, list)
            and index < len(action_reprs)
            and isinstance(action_reprs[index], str)
            else f"{normalized_op} target {target_id}"
        )
        messages.append(
            Message(
                role=Role.tool,
                tool_response=f"Observed source action: {action_repr}",
            )
        )
        capabilities.add(name)
        slot_values.setdefault("browser.target_id", []).append(target_id)
        if value:
            slot_values.setdefault("browser.operation_value", []).append(value)
    domain = raw.get("domain", "web")
    if not isinstance(domain, str) or not domain:
        domain = "web"
    return _NormalizedRecord(
        record_id=record_id,
        domain=domain,
        behavior="action",
        capabilities=tuple(sorted(capabilities)),
        slot_values={
            key: tuple(values) for key, values in sorted(slot_values.items())
        },
        tools=_mind2web_tools(),
        messages=tuple(messages),
        quality={"source_trace": "crowdsourced_action_sequence"},
        source_line=source_line,
    )


def _tool_spec(raw: object, *, label: str) -> ToolSpec:
    if not isinstance(raw, dict) or set(raw) != {"name", "description", "parameters"}:
        raise ValueError(f"{label} must contain exactly name, description, parameters")
    if not isinstance(raw["name"], str) or not isinstance(raw["description"], str):
        raise TypeError(f"{label} name and description must be text")
    parameters = _canonical_json_value(raw["parameters"], label=f"{label}.parameters")
    if not isinstance(parameters, dict):
        raise TypeError(f"{label}.parameters must be an object")
    return ToolSpec(raw["name"], raw["description"], parameters)


def _message(raw: object, *, label: str) -> Message:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be an object")
    allowed = {"role", "content", "tool_calls", "tool_response"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ValueError(f"{label} has unknown keys: {extra}")
    role = raw.get("role")
    try:
        parsed_role = Role(role)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.role is invalid") from error
    content = raw.get("content", "")
    response = raw.get("tool_response")
    calls_raw = raw.get("tool_calls", [])
    if not isinstance(content, str):
        raise TypeError(f"{label}.content must be text")
    if response is not None and not isinstance(response, str):
        raise TypeError(f"{label}.tool_response must be text or null")
    if not isinstance(calls_raw, list):
        raise TypeError(f"{label}.tool_calls must be a list")
    calls: list[ToolCall] = []
    for index, call in enumerate(calls_raw):
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError(f"{label}.tool_calls[{index}] is malformed")
        if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise TypeError(f"{label}.tool_calls[{index}] has invalid types")
        calls.append(
            ToolCall(
                call["name"],
                _canonical_json_value(
                    call["arguments"],
                    label=f"{label}.tool_calls[{index}].arguments",
                ),
            )
        )
    return Message(parsed_role, content=content, tool_calls=calls, tool_response=response)


def _localagent_record(raw: object, *, source_line: int) -> _NormalizedRecord:
    if not isinstance(raw, dict):
        raise TypeError(f"localagent record {source_line} must be an object")
    required = {
        "record_id",
        "domain",
        "behavior",
        "capabilities",
        "slot_values",
        "tools",
        "messages",
    }
    allowed = required | {"quality"}
    missing = sorted(required - set(raw))
    extra = sorted(set(raw) - allowed)
    if missing or extra:
        raise ValueError(
            f"localagent record {source_line} keys mismatch: missing={missing}, extra={extra}"
        )
    record_id = raw["record_id"]
    domain = raw["domain"]
    behavior = raw["behavior"]
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"localagent record {source_line}.record_id must be non-empty")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"localagent record {source_line}.domain must be non-empty")
    if behavior not in _BEHAVIORS:
        raise ValueError(f"localagent record {source_line}.behavior is invalid")
    capabilities_raw = raw["capabilities"]
    if (
        not isinstance(capabilities_raw, list)
        or not all(isinstance(value, str) and value for value in capabilities_raw)
    ):
        raise TypeError(f"localagent record {source_line}.capabilities must be text values")
    slots_raw = raw["slot_values"]
    if not isinstance(slots_raw, dict):
        raise TypeError(f"localagent record {source_line}.slot_values must be an object")
    slots: dict[str, tuple[Any, ...]] = {}
    for key in sorted(slots_raw):
        values = slots_raw[key]
        if not isinstance(key, str) or not key or not isinstance(values, list):
            raise TypeError(f"localagent record {source_line}.slot_values entries are invalid")
        slots[key] = tuple(
            _canonical_json_value(
                value,
                label=f"localagent record {source_line}.slot_values.{key}",
            )
            for value in values
        )
    tools_raw = raw["tools"]
    messages_raw = raw["messages"]
    if not isinstance(tools_raw, list) or not isinstance(messages_raw, list):
        raise TypeError(f"localagent record {source_line} tools/messages must be lists")
    tools = tuple(
        sorted(
            (
                _tool_spec(tool, label=f"localagent record {source_line}.tools[{index}]")
                for index, tool in enumerate(tools_raw)
            ),
            key=lambda tool: tool.name,
        )
    )
    messages = tuple(
        _message(message, label=f"localagent record {source_line}.messages[{index}]")
        for index, message in enumerate(messages_raw)
    )
    quality = raw.get("quality", {})
    if not isinstance(quality, dict):
        raise TypeError(f"localagent record {source_line}.quality must be an object")
    return _NormalizedRecord(
        record_id=record_id,
        domain=domain,
        behavior=behavior,
        capabilities=tuple(sorted(set(capabilities_raw))),
        slot_values=slots,
        tools=tools,
        messages=messages,
        quality=_canonical_json_value(
            quality,
            label=f"localagent record {source_line}.quality",
        ),
        source_line=source_line,
    )


def _validate_sequence(record: _NormalizedRecord) -> None:
    if len(record.messages) < 2:
        raise ValueError(f"record {record.record_id!r} must contain at least two messages")
    if not any(message.role == Role.user for message in record.messages):
        raise ValueError(f"record {record.record_id!r} has no user turn")
    if not any(message.role == Role.assistant for message in record.messages):
        raise ValueError(f"record {record.record_id!r} has no assistant decision")
    first_non_system = next(
        (message for message in record.messages if message.role != Role.system),
        None,
    )
    if first_non_system is None or first_non_system.role != Role.user:
        raise ValueError(f"record {record.record_id!r} must start with a user turn")
    pending_tool_responses = 0
    action_count = 0
    for index, message in enumerate(record.messages):
        if message.role == Role.assistant:
            if pending_tool_responses:
                raise ValueError(
                    f"record {record.record_id!r} leaves tool responses unresolved "
                    f"before message {index}"
                )
            pending_tool_responses = len(message.tool_calls)
            action_count += pending_tool_responses
        elif message.role == Role.tool:
            if pending_tool_responses < 1:
                raise ValueError(
                    f"record {record.record_id!r} message {index} has an orphan tool response"
                )
            pending_tool_responses -= 1
        elif pending_tool_responses:
            raise ValueError(
                f"record {record.record_id!r} leaves tool responses unresolved before message {index}"
            )
    if record.behavior == "action" and action_count < 1:
        raise ValueError(f"action record {record.record_id!r} has no tool call")
    if record.behavior != "action" and action_count:
        raise ValueError(f"{record.behavior} record {record.record_id!r} must not call tools")
    if record.behavior != "action":
        assistant_text = [
            message.content
            for message in record.messages
            if message.role == Role.assistant and message.content
        ]
        if not assistant_text:
            raise ValueError(f"{record.behavior} record {record.record_id!r} needs a text response")


def _conversation(
    record: _NormalizedRecord,
    source: PublicSourceSnapshot,
    *,
    enrichment_level: int,
    derivation: str,
    messages: Sequence[Message] | None = None,
    behavior: str | None = None,
) -> Conversation:
    selected_messages = tuple(messages) if messages is not None else record.messages
    action_count = sum(
        len(message.tool_calls)
        for message in selected_messages
        if message.role == Role.assistant
    )
    meta = {
        "category": record.domain,
        "group": "public_agent",
        "kind": "public_agent_trace",
        "split": source.split,
        "generator": PUBLIC_AGENT_GENERATOR,
        "public_data": True,
        "behavior": behavior or record.behavior,
        "capabilities": list(record.capabilities),
        "action_count": action_count,
        "enrichment_level": enrichment_level,
        "parent_record_id": record.record_id,
        "derivation": derivation,
        "quality": dict(record.quality),
        "slot_values": {
            key: list(values) for key, values in sorted(record.slot_values.items())
        },
        "provenance": source.provenance(
            record_id=record.record_id,
            source_line=record.source_line,
        ),
        "verified": False,
        "rule_verified": True,
        "model_verified": False,
        "environment_executed": False,
        "verification_scope": VERIFICATION_SCOPE,
    }
    conversation = Conversation(
        messages=list(selected_messages),
        tools=list(record.tools),
        meta=meta,
    )
    # This validates catalog names, JSON schemas, argument types, finite values, and canonical
    # tool-call renderability.  The renderer is the authority used by SFT and eval.
    assistant_training_turns(conversation)
    return conversation


def _replace_first_user(messages: Sequence[Message], content: str) -> list[Message]:
    result: list[Message] = []
    replaced = False
    for message in messages:
        if message.role == Role.user and not replaced:
            result.append(
                Message(
                    role=Role.user,
                    content=content,
                    tool_calls=list(message.tool_calls),
                    tool_response=message.tool_response,
                )
            )
            replaced = True
        else:
            result.append(
                Message(
                    role=message.role,
                    content=message.content,
                    tool_calls=[
                        ToolCall(call.name, dict(call.arguments)) for call in message.tool_calls
                    ],
                    tool_response=message.tool_response,
                )
            )
    if not replaced:  # pragma: no cover - _validate_sequence already proves a user turn
        raise AssertionError("validated record lost its first user turn")
    return result


def _enrich_train_record(
    record: _NormalizedRecord,
    source: PublicSourceSnapshot,
    *,
    max_level: int,
    seed: int,
) -> list[Conversation]:
    conversations = [
        _conversation(
            record,
            source,
            enrichment_level=0,
            derivation="source",
        )
    ]
    if source.split != "train" or max_level < 1:
        return conversations
    first_user = next(
        message.content for message in record.messages if message.role == Role.user
    )
    wrapper = _TRAIN_WRAPPERS[
        _stable_index(seed, source.source_id, record.record_id, 1, modulo=len(_TRAIN_WRAPPERS))
    ]
    conversations.append(
        _conversation(
            record,
            source,
            enrichment_level=1,
            derivation="operational_wrapper_v1",
            messages=_replace_first_user(record.messages, wrapper.format(request=first_user)),
        )
    )
    if max_level >= 2:
        wrapper = _TRAIN_CONTEXT_WRAPPERS[
            _stable_index(
                seed,
                source.source_id,
                record.record_id,
                2,
                modulo=len(_TRAIN_CONTEXT_WRAPPERS),
            )
        ]
        conversations.append(
            _conversation(
                record,
                source,
                enrichment_level=2,
                derivation="domain_context_wrapper_v1",
                messages=_replace_first_user(
                    record.messages,
                    wrapper.format(
                        domain=record.domain.replace("_", " "),
                        request=first_user,
                    ),
                ),
            )
        )
    if max_level >= 3 and record.behavior == "action":
        cancelled = (
            "The following request is cancelled and quoted only as background: "
            f'"{first_user}" Do not call any tool. Acknowledge the cancellation.'
        )
        conversations.append(
            _conversation(
                record,
                source,
                enrichment_level=3,
                derivation="counterfactual_cancel_v1",
                behavior="irrelevance",
                messages=[
                    Message(role=Role.user, content=cancelled),
                    Message(
                        role=Role.assistant,
                        content="Acknowledged. I will not take action.",
                    ),
                ],
            )
        )
    return conversations


def _iter_source_records(
    source: PublicSourceSnapshot,
    *,
    max_source_bytes: int,
) -> Iterator[_NormalizedRecord]:
    with _verified_text_source(source, max_source_bytes=max_source_bytes) as handle:
        if source.adapter == "localagent_v1":
            emitted = 0
            for source_line, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > _MAX_JSONL_ROW_BYTES:
                    raise ValueError(
                        f"source {source.source_id!r} line {source_line} exceeds row byte cap"
                    )
                raw = _strict_json_loads(
                    line,
                    label=f"source {source.source_id!r} line {source_line}",
                )
                yield _localagent_record(raw, source_line=source_line)
                emitted += 1
                if source.max_records is not None and emitted >= source.max_records:
                    return
            return
        if source.adapter == "agentnet_v1":
            emitted = 0
            for source_line, raw in _iter_jsonl(
                handle,
                label=f"source {source.source_id!r}",
            ):
                normalized = normalize_agentnet_record(
                    raw,
                    source_revision=source.revision,
                    split=source.split,
                )
                yield _localagent_record(normalized, source_line=source_line)
                emitted += 1
                if source.max_records is not None and emitted >= source.max_records:
                    return
            return
        for source_line, raw in _iter_json_array(
            handle,
            label=f"source {source.source_id!r}",
        ):
            if source.adapter == "xlam_v1":
                yield _xlam_record(raw, source_line=source_line)
            elif source.adapter == "mind2web_v1":
                yield _mind2web_record(
                    raw,
                    source_line=source_line,
                    max_actions=source.max_actions_per_record,
                )
            else:  # pragma: no cover - config validation constrains adapters
                raise AssertionError(source.adapter)
            if source.max_records is not None and source_line >= source.max_records:
                return


def _first_user_prompt(conversation: Conversation) -> str:
    return next(
        message.content
        for message in conversation.messages
        if message.role == Role.user
    )


def _slot_fingerprint(value: Any) -> str:
    if isinstance(value, str):
        normalized: Any = {"type": "string", "value": _normalized_text(value)}
    else:
        normalized = {
            "type": type(value).__name__,
            "value": _canonical_json_value(value, label="slot value"),
        }
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _split_audit(
    conversations: Mapping[str, Sequence[Conversation]],
) -> dict[str, Any]:
    slots: dict[str, dict[str, set[str]]] = {
        split: {} for split in _SPLITS
    }
    global_slots: dict[str, set[str]] = {split: set() for split in _SPLITS}
    prompts: dict[str, set[str]] = {split: set() for split in _SPLITS}
    semantics: dict[str, set[str]] = {split: set() for split in _SPLITS}
    for split in _SPLITS:
        for conversation in conversations.get(split, ()):
            prompts[split].add(_normalized_text(_first_user_prompt(conversation)))
            semantics[split].add(conversation_semantic_sha256(conversation))
            raw_slots = conversation.meta.get("slot_values", {})
            if not isinstance(raw_slots, dict):
                raise TypeError("public conversation slot_values metadata must be an object")
            for name, values in raw_slots.items():
                if not isinstance(name, str) or not isinstance(values, list):
                    raise TypeError("public conversation slot_values metadata is malformed")
                fingerprints = slots[split].setdefault(name, set())
                for value in values:
                    fingerprint = _slot_fingerprint(value)
                    fingerprints.add(fingerprint)
                    global_slots[split].add(fingerprint)
    prompt_overlap = prompts["train"] & prompts["eval"]
    semantic_overlap = semantics["train"] & semantics["eval"]
    slot_overlap = global_slots["train"] & global_slots["eval"]
    if prompt_overlap:
        raise ValueError("public train/eval first-user prompts must be disjoint")
    if semantic_overlap:
        raise ValueError("public train/eval semantic conversations must be disjoint")
    if slot_overlap:
        raise ValueError("public train/eval declared slot values must be disjoint")
    return {
        "contract": (
            "NFKC-casefolded first-user prompts, canonical semantic rows, and globally declared "
            "typed slot values are disjoint across train/eval"
        ),
        "paired_splits_present": bool(conversations.get("train")) and bool(
            conversations.get("eval")
        ),
        "prompt_counts": {split: len(prompts[split]) for split in sorted(_SPLITS)},
        "semantic_counts": {split: len(semantics[split]) for split in sorted(_SPLITS)},
        "slot_counts": {split: len(global_slots[split]) for split in sorted(_SPLITS)},
        "per_slot_counts": {
            split: {
                name: len(values) for name, values in sorted(slots[split].items())
            }
            for split in sorted(_SPLITS)
        },
        "prompt_overlap": 0,
        "semantic_overlap": 0,
        "slot_overlap": 0,
    }


def _holdout_prompts(
    raw: object,
    *,
    config_path: Path,
) -> tuple[set[str], list[dict[str, object]]]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise TypeError("exact_prompt_holdouts must be a list")
    prompts: set[str] = set()
    artifacts: list[dict[str, object]] = []
    names: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"name", "path", "bytes", "sha256"}:
            raise ValueError(
                f"exact_prompt_holdouts[{index}] must contain name/path/bytes/sha256"
            )
        name = entry["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"exact_prompt_holdouts[{index}].name is invalid or duplicate")
        names.add(name)
        declared_path, path = _file_path(
            config_path.parent,
            entry["path"],
            label=f"exact_prompt_holdouts[{index}].path",
        )
        expected_bytes = _non_negative_int(
            entry["bytes"],
            label=f"exact_prompt_holdouts[{index}].bytes",
        )
        expected_sha = entry["sha256"]
        if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError(f"exact_prompt_holdouts[{index}].sha256 is invalid")
        payload = path.read_bytes()
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"exact prompt holdout {name!r} identity mismatch")
        normalized = {_normalized_text(prompt) for prompt in read_evaluation_denylist(path)}
        if not normalized or "" in normalized:
            raise ValueError(f"exact prompt holdout {name!r} contains no usable prompts")
        prompts.update(normalized)
        artifacts.append(
            {
                "name": name,
                "path": declared_path,
                "bytes": expected_bytes,
                "sha256": expected_sha,
                "normalized_entries": len(normalized),
            }
        )
    return prompts, sorted(artifacts, key=lambda item: str(item["name"]))


def _publish_jsonl(path: Path, conversations: Sequence[Conversation]) -> FileIdentity:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for conversation in conversations:
                payload = (conversation.to_json() + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
                byte_count += len(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return FileIdentity(bytes=byte_count, sha256=digest.hexdigest())


def _count_rows(
    conversations: Mapping[str, Sequence[Conversation]],
) -> dict[str, Any]:
    split_counts: dict[str, Any] = {}
    for split in sorted(conversations):
        rows = conversations[split]
        behavior = Counter(str(row.meta["behavior"]) for row in rows)
        domains = Counter(str(row.meta["category"]) for row in rows)
        derivations = Counter(str(row.meta["derivation"]) for row in rows)
        capabilities = Counter(
            capability
            for row in rows
            for capability in row.meta.get("capabilities", [])
        )
        action_turn_counts = [
            sum(
                message.role == Role.assistant and bool(message.tool_calls)
                for message in row.messages
            )
            for row in rows
        ]
        multi_turn_action_rows = sum(count > 1 for count in action_turn_counts)
        parallel_action_rows = sum(
            any(
                message.role == Role.assistant and len(message.tool_calls) > 1
                for message in row.messages
            )
            for row in rows
        )
        split_counts[split] = {
            "rows": len(rows),
            "assistant_decisions": sum(
                sum(message.role == Role.assistant for message in row.messages)
                for row in rows
            ),
            "tool_calls": sum(int(row.meta["action_count"]) for row in rows),
            "multi_step_rows": multi_turn_action_rows,
            "multi_turn_action_rows": multi_turn_action_rows,
            "parallel_action_rows": parallel_action_rows,
            "behavior": dict(sorted(behavior.items())),
            "domains": dict(sorted(domains.items())),
            "derivations": dict(sorted(derivations.items())),
            "capabilities": dict(sorted(capabilities.items())),
        }
    return split_counts


def build_public_agent_dataset(config_path: str | Path) -> PublicAgentBuildResult:
    """Normalize pinned public snapshots and publish deterministic split JSONL + audit manifest."""

    config_file = Path(config_path).resolve()
    payload = config_file.read_bytes()
    if len(payload) > _MAX_CONFIG_BYTES:
        raise ValueError("public-agent config exceeds the hard byte cap")
    try:
        config = yaml.safe_load(payload.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as error:
        raise ValueError("public-agent config must be UTF-8") from error
    if not isinstance(config, dict):
        raise TypeError("public-agent config must be a mapping")
    allowed = {
        "schema_version",
        "seed",
        "enrichment_level",
        "max_source_bytes",
        "outputs",
        "manifest",
        "sources",
        "exact_prompt_holdouts",
    }
    extra = sorted(set(config) - allowed)
    if extra:
        raise ValueError(f"public-agent config has unknown keys: {extra}")
    if config.get("schema_version") != PUBLIC_AGENT_SCHEMA_VERSION:
        raise ValueError(
            f"public-agent config schema_version must be {PUBLIC_AGENT_SCHEMA_VERSION}"
        )
    seed = _non_negative_int(config.get("seed", 0), label="seed")
    enrichment_level = _non_negative_int(
        config.get("enrichment_level", 0),
        label="enrichment_level",
    )
    if enrichment_level > 3:
        raise ValueError("enrichment_level must be in [0, 3]")
    max_source_bytes = _non_negative_int(
        config.get("max_source_bytes", _DEFAULT_MAX_SOURCE_BYTES),
        label="max_source_bytes",
    )
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("public-agent config requires at least one source")
    sources = [
        _source_from_config(raw, index=index, base=config_file.parent)
        for index, raw in enumerate(raw_sources)
    ]
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("public-agent source_id values must be unique")
    sources.sort(
        key=lambda source: (
            source.split,
            source.dataset,
            source.subset,
            source.revision,
            source.source_id,
        )
    )
    raw_outputs = config.get("outputs")
    if not isinstance(raw_outputs, dict) or not raw_outputs:
        raise ValueError("outputs must map at least one split to a path")
    if not set(raw_outputs).issubset(_SPLITS):
        raise ValueError("outputs keys must be train and/or eval")
    outputs: dict[str, Path] = {}
    declared_outputs: dict[str, str] = {}
    for split, raw_path in sorted(raw_outputs.items()):
        declared, path = _file_path(
            config_file.parent,
            raw_path,
            label=f"outputs.{split}",
        )
        outputs[split] = path
        declared_outputs[split] = declared
    manifest_declared, manifest_path = _file_path(
        config_file.parent,
        config.get("manifest"),
        label="manifest",
    )
    all_output_paths = [*outputs.values(), manifest_path]
    if len(all_output_paths) != len(set(all_output_paths)):
        raise ValueError("output and manifest paths must be distinct")
    if any(source.path in all_output_paths for source in sources):
        raise ValueError("public source and output paths must be distinct")
    source_splits = {source.split for source in sources}
    if not source_splits.issubset(outputs):
        raise ValueError("every configured source split needs an output path")

    denylist, holdout_artifacts = _holdout_prompts(
        config.get("exact_prompt_holdouts", []),
        config_path=config_file,
    )
    rows: dict[str, list[Conversation]] = {split: [] for split in outputs}
    record_ids: set[tuple[str, str, str, str]] = set()
    for source in sources:
        for record in _iter_source_records(source, max_source_bytes=max_source_bytes):
            identity = (
                source.dataset,
                source.subset,
                source.revision,
                record.record_id,
            )
            if identity in record_ids:
                raise ValueError(f"duplicate public record identity: {identity}")
            record_ids.add(identity)
            _validate_sequence(record)
            base_prompt = next(
                message.content for message in record.messages if message.role == Role.user
            )
            if _normalized_text(base_prompt) in denylist:
                raise ValueError(
                    f"public record {record.record_id!r} matches an exact eval prompt holdout"
                )
            rows[source.split].extend(
                _enrich_train_record(
                    record,
                    source,
                    max_level=enrichment_level,
                    seed=seed,
                )
            )
    for split, split_rows in rows.items():
        if not split_rows:
            raise ValueError(f"configured output split {split!r} produced no rows")
        split_rows.sort(
            key=lambda row: (
                str(row.meta["provenance"]["dataset"]),
                str(row.meta["provenance"]["subset"]),
                str(row.meta["provenance"]["revision"]),
                str(row.meta["parent_record_id"]),
                int(row.meta["enrichment_level"]),
                str(row.meta["derivation"]),
            )
        )
    split_audit = _split_audit(rows)
    output_identities = {
        split: _publish_jsonl(outputs[split], rows[split])
        for split in sorted(rows)
    }
    config_identity = FileIdentity.from_bytes(payload)
    counts = _count_rows(rows)
    source_manifest = [
        {
            **{
                key: value
                for key, value in asdict(source).items()
                if key not in {"path"}
            },
            "path": source.declared_path,
        }
        for source in sources
    ]
    manifest_core = {
        "kind": PUBLIC_AGENT_ARTIFACT_KIND,
        "schema_version": PUBLIC_AGENT_SCHEMA_VERSION,
        "format": "localagent.data.schema.Conversation",
        "generator": PUBLIC_AGENT_GENERATOR,
        "generator_config": config_identity.as_dict(),
        "seed": seed,
        "enrichment_level": enrichment_level,
        "verification": {
            "rule_verified": True,
            "model_verified": False,
            "environment_executed": False,
            "scope": VERIFICATION_SCOPE,
        },
        "sources": source_manifest,
        "licenses": dict(sorted(Counter(source.license for source in sources).items())),
        "outputs": {
            split: {
                "path": declared_outputs[split],
                **output_identities[split].as_dict(),
                **counts[split],
            }
            for split in sorted(rows)
        },
        "manifest_path": manifest_declared,
        "split_audit": split_audit,
        "exact_prompt_holdouts": {
            "artifacts": holdout_artifacts,
            "normalized_unique_prompts": len(denylist),
            "matches": 0,
        },
        "boundary_policy": {
            "xlam": "pinned official train revision only",
            "mind2web": "pinned data/train/train_*.json only; benchmark test data rejected",
            "bfcl": "benchmark material rejected for train",
            "weblinx": "eval-only/non-default; noncommercial data rejected for train",
        },
    }
    manifest, manifest_payload = self_hashed_manifest(manifest_core)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(manifest_payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return PublicAgentBuildResult(
        conversations={
            split: tuple(split_rows) for split, split_rows in rows.items()
        },
        outputs=outputs,
        manifest_path=manifest_path,
        manifest=manifest,
    )
