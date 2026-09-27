"""Freeze and audit a fresh external action-evaluation slice.

This module deliberately does not download benchmark data.  A benchmark adapter must first
produce the explicit, revision-bound JSON format documented in
``docs/paper/FRESH_EXTERNAL_EVAL_CONTRACT.md``.  The freeze then:

* verifies the exact source and every declared training artifact by byte size and SHA-256;
* selects cases with a deterministic, bounded algorithm;
* assigns content-bound case IDs and revision-namespaced cluster/template grouping IDs;
* fails closed on prompt/shingle overlap with declared stage text (including RL in v2);
* fails closed on derived action-template overlap with labeled Conversation training rows; and
* publishes a frozen evaluation bundle, prompt-only denylist, and self-hashed manifest.

V1 remains the historical normalized-call lane.  V2 additionally binds exact full-catalog
conversation rendering and the pretrain -> midtrain -> SFT -> RL lineage chain.  The output is
evaluation-only.  The prompt-only denylist is the sole output intended for future corpus
preparation; expected calls must never be fed back into training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localagent.data.pretrain_corpus import CorpusDocument, screen_evaluation_contamination
from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    assistant_training_examples,
    render_agent_decode_prompt,
    schema_matches,
    validate_tool_catalog,
)
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.model.tokenizer import BPE_EOS

CONTRACT_KIND = "localagent_fresh_external_action_eval_contract"
SOURCE_KIND = "localagent_external_action_export"
SLICE_KIND = "localagent_frozen_external_action_slice"
MANIFEST_KIND = "localagent_fresh_external_action_eval_manifest"
DENYLIST_KIND = "localagent_evaluation_prompt_denylist"
TRAINING_LINEAGE_KIND = "localagent_training_lineage_export"
SCHEMA_VERSION = 1
HARDENED_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, HARDENED_SCHEMA_VERSION})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT_SHA256 = re.compile(r"[0-9a-f]{24}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_LEGACY_STAGES = frozenset({"pretrain", "midtrain", "sft"})
_HARDENED_STAGES = frozenset({"pretrain", "midtrain", "sft", "rl"})
_FORMATS = frozenset({"conversation_jsonl", "corpus_jsonl", "text"})
_DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_DEFAULT_MAX_RECORD_BYTES = 8 * 1024 * 1024
_MAX_CONTRACT_BYTES = 4 * 1024 * 1024
_MAX_BOOTSTRAP_RESAMPLES = 1_000_000
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_CASES = 50_000
_AUDIT_CHUNK_ROWS = 512
_AUDIT_CHUNK_CHARS = 16 * 1024 * 1024

_SOURCE_KEYS = frozenset({"kind", "schema_version", "benchmark", "revision", "split", "cases"})
_SOURCE_CASE_KEYS = frozenset(
    {
        "source_case_id",
        "task_cluster_id",
        "template_id",
        "family",
        "prompt",
        "tools",
        "expected_calls",
        "metadata",
    }
)
_TOOL_KEYS = frozenset({"name", "description", "parameters"})
_CALL_KEYS = frozenset({"name", "arguments"})
_CONTRACT_KEYS_V1 = frozenset(
    {
        "kind",
        "schema_version",
        "source",
        "limits",
        "selection",
        "decontamination",
        "training_artifacts",
        "analysis",
    }
)
_CONTRACT_KEYS_V2 = _CONTRACT_KEYS_V1 | {"lineage_artifacts"}
_SOURCE_DECLARATION_KEYS = frozenset({"path", "bytes", "sha256", "benchmark", "revision", "split"})
_LIMIT_KEYS = frozenset(
    {
        "max_artifact_bytes",
        "max_source_bytes",
        "max_record_bytes",
        "max_source_cases",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "seed",
        "min_cases",
        "max_cases",
        "min_task_clusters",
        "max_cases_per_task_cluster",
        "max_cases_per_template",
    }
)
_DECONTAMINATION_KEYS = frozenset(
    {
        "shingle_size",
        "min_shingles",
        "min_coverage",
        "anchors_per_entry",
        "max_denylist_shingles",
    }
)
_TRAINING_ARTIFACT_KEYS_V1 = frozenset(
    {"stage", "name", "format", "path", "records", "bytes", "sha256"}
)
_TRAINING_ARTIFACT_KEYS_V2_CORPUS = _TRAINING_ARTIFACT_KEYS_V1
_TRAINING_ARTIFACT_KEYS_V2_CONVERSATION = _TRAINING_ARTIFACT_KEYS_V1 | {
    "conversation_prompt_contract"
}
_LINEAGE_DECLARATION_KEYS = frozenset({"stage", "name", "path", "bytes", "sha256"})
_LINEAGE_EXPORT_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "stage",
        "checkpoint_sha256",
        "lineage",
        "training_artifact_sha256",
        "conversation_prompt_contract",
    }
)
_STAGE_LINEAGE_REQUIRED_KEYS = frozenset(
    {
        "version",
        "stage",
        "config_sha256",
        "model_config_sha256",
        "data_sha256",
        "tokenizer_sha256",
        "git",
    }
)
_STAGE_LINEAGE_OPTIONAL_KEYS = frozenset({"parent_checkpoint_sha256"})
_STAGE_GIT_KEYS = frozenset({"commit", "repository_sha256", "dirty", "worktree_sha256"})
_PARENT_STAGE = {"midtrain": "pretrain", "sft": "midtrain", "rl": "sft"}
_ANALYSIS_KEYS = frozenset(
    {
        "bootstrap_resamples",
        "bootstrap_seed",
        "exact_action_noninferiority_margin",
    }
)
_SLICE_KEYS = frozenset({"kind", "schema_version", "benchmark", "revision", "split", "cases"})
_FROZEN_CASE_KEYS = frozenset(
    {
        "case_id",
        "task_cluster_id",
        "template_id",
        "derived_template_sha256",
        "source_identity",
        "family",
        "prompt",
        "tools",
        "expected_calls",
        "metadata",
    }
)
_SOURCE_IDENTITY_KEYS = frozenset(
    {
        "source_case_id_sha256",
        "source_task_cluster_id_sha256",
        "source_template_id_sha256",
        "source_index",
    }
)
_CONVERSATION_KEYS = frozenset({"messages", "tools", "meta"})
_MESSAGE_KEYS = frozenset({"role", "content", "tool_calls", "tool_response"})


@dataclass(frozen=True)
class _ExternalCase:
    """Validated source case plus immutable public case and grouping identities."""

    source_index: int
    source_case_id: str
    source_cluster_id: str
    source_template_id: str
    case_id: str
    task_cluster_id: str
    template_id: str
    derived_template_sha256: str
    family: str
    prompt: str
    tools: list[dict[str, Any]]
    expected_calls: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _TrainingUnit:
    """One bounded piece of training text and any derivable action templates."""

    unit_id: str
    text: str
    template_sha256: frozenset[str]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_loads(payload: bytes | str, *, label: str) -> Any:
    """Decode finite JSON while rejecting duplicate object keys at every depth."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value!r}")

    try:
        text = payload.decode("utf-8", errors="strict") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid UTF-8 JSON") from error


def _exact_mapping(
    value: Any,
    keys: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _mapping_with_optional_keys(
    value: Any,
    required: frozenset[str],
    optional: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _file_sha256(path: Path, *, label: str, max_bytes: int) -> tuple[int, str]:
    """Hash one descriptor-bound, non-symlink regular file."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from error
    digest = hashlib.sha256()
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        if initial.st_size > max_bytes:
            raise ValueError(
                f"{label} exceeds max_artifact_bytes ({initial.st_size} > {max_bytes})"
            )
        try:
            path_state = path.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} pathname changed while being verified: {path}") from error
        if not _same_file_state(initial, path_state):
            raise RuntimeError(f"{label} changed while its descriptor was being bound: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        final = os.fstat(descriptor)
        try:
            final_path = path.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} pathname changed while being verified: {path}") from error
        if not _same_file_state(initial, final) or not _same_file_state(initial, final_path):
            raise RuntimeError(f"{label} changed while it was being checksummed: {path}")
        return initial.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, label: str, max_bytes: int) -> bytes:
    """Read one bounded descriptor-bound, non-symlink regular file."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        if initial.st_size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        path_state = path.lstat()
        if not _same_file_state(initial, path_state):
            raise RuntimeError(f"{label} changed while its descriptor was being bound: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        final = os.fstat(descriptor)
        final_path = path.lstat()
        if not _same_file_state(initial, final) or not _same_file_state(initial, final_path):
            raise RuntimeError(f"{label} changed while it was being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_fraction(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return result


def _declared_artifact(
    record: Mapping[str, Any],
    *,
    base: Path,
    name: str,
    max_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    raw_path = _nonempty_string(record.get("path"), name=f"{name}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    expected_bytes = _nonnegative_int(record.get("bytes"), name=f"{name}.bytes")
    expected_sha256 = record.get("sha256")
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256")
    observed_bytes, observed_sha256 = _file_sha256(
        path,
        label=name,
        max_bytes=max_bytes,
    )
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"{name} byte-size mismatch: expected {expected_bytes}, got {observed_bytes}"
        )
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    return path, {
        "path": raw_path,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def verify_declared_file_identity(
    record: Mapping[str, Any],
    *,
    base: str | Path,
    label: str,
    max_bytes: int = _MAX_ARTIFACT_BYTES,
) -> tuple[Path, dict[str, Any]]:
    """Verify one declared regular-file identity against the bytes actually on disk."""

    declaration = _exact_mapping(
        record,
        frozenset({"path", "bytes", "sha256"}),
        label=label,
    )
    return _declared_artifact(
        declaration,
        base=Path(base),
        name=label,
        max_bytes=max_bytes,
    )


def _normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def normalize_prompt(text: str) -> str:
    """Return the freeze contract's canonical prompt representation."""

    return " ".join(_normalized_tokens(text))


def _scalar_tokens(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        tokens = _normalized_tokens(value)
    elif value is True:
        tokens = ["true"]
    elif value is False:
        tokens = ["false"]
    elif value is None:
        tokens = ["null"]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("action arguments must not contain non-finite numbers")
        tokens = _normalized_tokens(json.dumps(value, allow_nan=False))
    else:
        return None
    return tuple(tokens) if tokens else None


def _argument_token_sequences(value: Any) -> list[tuple[str, ...]]:
    sequences: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            sequences.extend(_argument_token_sequences(value[key]))
    elif isinstance(value, list):
        for item in value:
            sequences.extend(_argument_token_sequences(item))
    else:
        tokens = _scalar_tokens(value)
        if tokens is not None:
            sequences.append(tokens)
    return sequences


def action_template_sha256(prompt: str, calls: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint a prompt skeleton after replacing gold scalar arguments.

    This is deliberately a conservative *derived skeleton* check, not a semantic proof that two
    natural-language templates are unrelated.  Tool names remain part of the fingerprint.
    """

    prompt_tokens = _normalized_tokens(prompt)
    replacements = sorted(
        {
            sequence
            for call in calls
            for sequence in _argument_token_sequences(call.get("arguments", {}))
            if sequence
        },
        key=lambda sequence: (-len(sequence), sequence),
    )
    skeleton: list[str] = []
    index = 0
    while index < len(prompt_tokens):
        matched = next(
            (
                sequence
                for sequence in replacements
                if tuple(prompt_tokens[index : index + len(sequence)]) == sequence
            ),
            None,
        )
        if matched is None:
            skeleton.append(prompt_tokens[index])
            index += 1
        else:
            skeleton.append("<arg>")
            index += len(matched)
    semantic = {
        "prompt_skeleton_tokens": skeleton,
        # Final AST scoring is call-order insensitive, so template screening must be too.
        "tool_names_multiset": sorted(str(call.get("name", "")) for call in calls),
    }
    return _sha256(_canonical_bytes(semantic))


def _content_id(prefix: str, *parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return f"{prefix}-{_sha256(payload)[:24]}"


def _validate_tool(tool: Any, *, name: str) -> dict[str, Any]:
    raw = _exact_mapping(tool, _TOOL_KEYS, label=name)
    tool_name = _nonempty_string(raw.get("name"), name=f"{name}.name")
    if _NAME.fullmatch(tool_name) is None:
        raise ValueError(f"{name}.name contains unsupported characters")
    description = raw.get("description")
    if not isinstance(description, str):
        raise TypeError(f"{name}.description must be a string")
    parameters = raw.get("parameters")
    spec = ToolSpec(name=tool_name, description=description, parameters=parameters)
    validate_tool_catalog([spec], label=name)
    return {
        "name": tool_name,
        "description": description,
        "parameters": parameters,
    }


def _validate_call(
    call: Any,
    *,
    tools: Mapping[str, dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    raw = _exact_mapping(call, _CALL_KEYS, label=name)
    call_name = _nonempty_string(raw.get("name"), name=f"{name}.name")
    arguments = raw.get("arguments")
    if not isinstance(arguments, dict):
        raise TypeError(f"{name}.arguments must be an object")
    spec = tools.get(call_name)
    if spec is None:
        raise ValueError(f"{name} names undeclared tool {call_name!r}")
    if not schema_matches(arguments, spec["parameters"]):
        raise ValueError(f"{name}.arguments fail the declared tool schema")
    return {"name": call_name, "arguments": arguments}


def _load_source_cases(
    source_payload: bytes,
    *,
    source_label: str,
    expected_benchmark: str,
    expected_revision: str,
    expected_split: str,
    max_cases: int,
    schema_version: int,
) -> list[_ExternalCase]:
    source = _exact_mapping(
        _strict_json_loads(source_payload, label=source_label),
        _SOURCE_KEYS,
        label="external source export",
    )
    if source.get("kind") != SOURCE_KIND or source.get("schema_version") != schema_version:
        raise ValueError(f"external source must be {SOURCE_KIND!r} schema_version {schema_version}")
    benchmark = _nonempty_string(source.get("benchmark"), name="source.benchmark")
    revision = _nonempty_string(source.get("revision"), name="source.revision")
    split = _nonempty_string(source.get("split"), name="source.split")
    if (benchmark, revision, split) != (
        expected_benchmark,
        expected_revision,
        expected_split,
    ):
        raise ValueError(
            "source benchmark/revision/split disagree with the externally timestamped contract"
        )
    cases = source.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source.cases must be a non-empty array")
    if len(cases) > max_cases:
        raise ValueError(f"source contains {len(cases)} cases, exceeding max_source_cases")

    parsed: list[_ExternalCase] = []
    source_ids: set[str] = set()
    normalized_prompts: set[str] = set()
    for index, raw_case in enumerate(cases):
        label = f"source.cases[{index}]"
        raw_case = _exact_mapping(raw_case, _SOURCE_CASE_KEYS, label=label)
        source_case_id = _nonempty_string(
            raw_case.get("source_case_id"), name=f"{label}.source_case_id"
        )
        source_cluster_id = _nonempty_string(
            raw_case.get("task_cluster_id"), name=f"{label}.task_cluster_id"
        )
        source_template_id = _nonempty_string(
            raw_case.get("template_id"), name=f"{label}.template_id"
        )
        family = _nonempty_string(raw_case.get("family"), name=f"{label}.family")
        prompt = _nonempty_string(raw_case.get("prompt"), name=f"{label}.prompt")
        if source_case_id in source_ids:
            raise ValueError(f"duplicate source_case_id {source_case_id!r}")
        source_ids.add(source_case_id)
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            raise ValueError(f"{label}.prompt is empty after normalization")
        if normalized_prompt in normalized_prompts:
            raise ValueError(f"{label}.prompt duplicates another normalized prompt")
        normalized_prompts.add(normalized_prompt)

        raw_tools = raw_case.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValueError(f"{label}.tools must be a non-empty array")
        tools = [
            _validate_tool(tool, name=f"{label}.tools[{tool_index}]")
            for tool_index, tool in enumerate(raw_tools)
        ]
        tools_by_name = {tool["name"]: tool for tool in tools}
        if len(tools_by_name) != len(tools):
            raise ValueError(f"{label}.tools contains duplicate names")
        tool_specs = [
            ToolSpec(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
            )
            for tool in tools
        ]
        # This validates both the recursive catalog schema and prompt framing with the same helper
        # used by training/evaluation.  Reserved control markers therefore cannot counterfeit the
        # user/catalog boundary in a supposedly external case.
        render_agent_decode_prompt(
            [Message(role=Role.user, content=prompt)],
            tool_specs,
        )

        raw_calls = raw_case.get("expected_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError(f"{label}.expected_calls must be a non-empty array")
        expected_calls = [
            _validate_call(
                call,
                tools=tools_by_name,
                name=f"{label}.expected_calls[{call_index}]",
            )
            for call_index, call in enumerate(raw_calls)
        ]
        metadata = raw_case.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError(f"{label}.metadata must be an object")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.metadata must contain only finite JSON values") from error

        template_sha256 = action_template_sha256(prompt, expected_calls)
        semantic = {
            "benchmark": benchmark,
            "revision": revision,
            "split": split,
            "source_case_id": source_case_id,
            "source_task_cluster_id": source_cluster_id,
            "source_template_id": source_template_id,
            "family": family,
            "prompt": prompt,
            "tools": tools,
            "expected_calls": expected_calls,
            "metadata": metadata,
        }
        case_id = _content_id("extcase", _sha256(_canonical_bytes(semantic)))
        parsed.append(
            _ExternalCase(
                source_index=index,
                source_case_id=source_case_id,
                source_cluster_id=source_cluster_id,
                source_template_id=source_template_id,
                case_id=case_id,
                task_cluster_id=_content_id(
                    "extcluster", benchmark, revision, split, source_cluster_id
                ),
                template_id=_content_id(
                    "exttemplate",
                    benchmark,
                    revision,
                    split,
                    source_template_id,
                    template_sha256,
                ),
                derived_template_sha256=template_sha256,
                family=family,
                prompt=prompt,
                tools=tools,
                expected_calls=expected_calls,
                metadata=metadata,
            )
        )
    return parsed


def _select_cases(
    cases: Sequence[_ExternalCase],
    selection: Mapping[str, Any],
) -> tuple[list[_ExternalCase], dict[str, Any]]:
    seed = _nonempty_string(selection.get("seed"), name="selection.seed")
    min_cases = _positive_int(selection.get("min_cases"), name="selection.min_cases")
    max_cases = _positive_int(selection.get("max_cases"), name="selection.max_cases")
    min_clusters = _positive_int(
        selection.get("min_task_clusters"), name="selection.min_task_clusters"
    )
    if min_clusters < 2:
        raise ValueError("selection.min_task_clusters must be at least 2")
    max_per_cluster = _positive_int(
        selection.get("max_cases_per_task_cluster"),
        name="selection.max_cases_per_task_cluster",
    )
    max_per_template = _positive_int(
        selection.get("max_cases_per_template"), name="selection.max_cases_per_template"
    )
    if min_cases > max_cases:
        raise ValueError("selection.min_cases must be <= selection.max_cases")

    ranked = sorted(
        cases,
        key=lambda case: (
            _sha256(f"{seed}\0{case.case_id}".encode()),
            case.case_id,
        ),
    )
    cluster_counts: dict[str, int] = defaultdict(int)
    template_counts: dict[str, int] = defaultdict(int)
    selected: list[_ExternalCase] = []
    skipped_cluster_cap = 0
    skipped_template_cap = 0
    for case in ranked:
        if len(selected) == max_cases:
            break
        if cluster_counts[case.task_cluster_id] >= max_per_cluster:
            skipped_cluster_cap += 1
            continue
        if template_counts[case.template_id] >= max_per_template:
            skipped_template_cap += 1
            continue
        selected.append(case)
        cluster_counts[case.task_cluster_id] += 1
        template_counts[case.template_id] += 1
    if len(selected) < min_cases:
        raise ValueError(f"selection produced {len(selected)} cases, below min_cases={min_cases}")
    if len(cluster_counts) < min_clusters:
        raise ValueError(
            f"selection produced {len(cluster_counts)} task clusters, "
            f"below min_task_clusters={min_clusters}"
        )
    return sorted(selected, key=lambda case: case.case_id), {
        "algorithm": "sha256(seed_nul_content_bound_case_id)_rank_with_cluster_template_caps_v1",
        "seed": seed,
        "source_cases": len(cases),
        "selected_cases": len(selected),
        "selected_task_clusters": len(cluster_counts),
        "selected_templates": len(template_counts),
        "min_cases": min_cases,
        "max_cases": max_cases,
        "min_task_clusters": min_clusters,
        "max_cases_per_task_cluster": max_per_cluster,
        "max_cases_per_template": max_per_template,
        "skipped_by_task_cluster_cap": skipped_cluster_cap,
        "skipped_by_template_cap": skipped_template_cap,
        "selected_case_ids_sha256": _sha256(
            ("\n".join(case.case_id for case in selected) + "\n").encode("ascii")
        ),
    }


def _strict_conversation(raw: bytes, *, unit_prefix: str) -> Conversation:
    decoded = _exact_mapping(
        _strict_json_loads(raw, label=unit_prefix),
        _CONVERSATION_KEYS,
        label=unit_prefix,
    )
    raw_tools = decoded["tools"]
    raw_messages = decoded["messages"]
    if not isinstance(raw_tools, list):
        raise TypeError(f"{unit_prefix}.tools must be an array")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"{unit_prefix}.messages must be a non-empty array")
    if not isinstance(decoded["meta"], dict):
        raise TypeError(f"{unit_prefix}.meta must be an object")

    tools = [
        ToolSpec(**_validate_tool(tool, name=f"{unit_prefix}.tools[{index}]"))
        for index, tool in enumerate(raw_tools)
    ]
    messages: list[Message] = []
    for index, value in enumerate(raw_messages):
        label = f"{unit_prefix}.messages[{index}]"
        message = _exact_mapping(value, _MESSAGE_KEYS, label=label)
        try:
            role = Role(message["role"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.role is unsupported") from error
        if not isinstance(message["content"], str):
            raise TypeError(f"{label}.content must be text")
        raw_calls = message["tool_calls"]
        if not isinstance(raw_calls, list):
            raise TypeError(f"{label}.tool_calls must be an array")
        calls = []
        for call_index, raw_call in enumerate(raw_calls):
            call = _exact_mapping(
                raw_call,
                _CALL_KEYS,
                label=f"{label}.tool_calls[{call_index}]",
            )
            if not isinstance(call["name"], str) or not call["name"]:
                raise ValueError(f"{label}.tool_calls[{call_index}].name must be non-empty text")
            if not isinstance(call["arguments"], dict):
                raise TypeError(f"{label}.tool_calls[{call_index}].arguments must be an object")
            calls.append(ToolCall(name=call["name"], arguments=call["arguments"]))
        tool_response = message["tool_response"]
        if tool_response is not None and not isinstance(tool_response, str):
            raise ValueError(f"{label}.tool_response must be text or null")
        messages.append(
            Message(
                role=role,
                content=message["content"],
                tool_calls=calls,
                tool_response=tool_response,
            )
        )
    conversation = Conversation(messages=messages, tools=tools, meta=decoded["meta"])
    return conversation


def _conversation_training_units(
    raw: bytes,
    *,
    unit_prefix: str,
    schema_version: int,
    conversation_prompt_contract: str | None,
) -> list[_TrainingUnit]:
    try:
        conversation = _strict_conversation(raw, unit_prefix=unit_prefix)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{unit_prefix} is not a valid Conversation row") from error

    messages = conversation.messages
    templates: set[str] = set()
    if schema_version == HARDENED_SCHEMA_VERSION:
        for index, message in enumerate(messages):
            if message.role != Role.assistant or not message.tool_calls:
                continue
            user = next(
                (
                    candidate
                    for candidate in reversed(messages[:index])
                    if candidate.role == Role.user
                ),
                None,
            )
            if user is not None:
                calls = [
                    {"name": call.name, "arguments": call.arguments} for call in message.tool_calls
                ]
                templates.add(action_template_sha256(user.content, calls))
    else:
        # Preserve the v1 template pairing algorithm exactly for historical replay.
        for index, message in enumerate(messages):
            if message.role != Role.user:
                continue
            next_assistant = next(
                (
                    candidate
                    for candidate in messages[index + 1 :]
                    if candidate.role in {Role.user, Role.assistant}
                ),
                None,
            )
            if next_assistant is not None and next_assistant.role == Role.assistant:
                calls = [
                    {"name": call.name, "arguments": call.arguments}
                    for call in next_assistant.tool_calls
                ]
                if calls:
                    templates.add(action_template_sha256(message.content, calls))

    if schema_version == HARDENED_SCHEMA_VERSION:
        if conversation_prompt_contract != OPENAI_FULL_CATALOG_V1:
            raise ValueError(
                f"{unit_prefix} hardened conversation audit supports only "
                f"conversation_prompt_contract={OPENAI_FULL_CATALOG_V1!r}; "
                "legacy/external renderers require a separately versioned adapter"
            )
        examples = assistant_training_examples(conversation)
        if not examples:
            raise ValueError(f"{unit_prefix} has no assistant training decisions")
        return [
            _TrainingUnit(
                unit_id=f"{unit_prefix}:assistant-{example.message_index}",
                # This is the exact full-catalog textual sequence encoded by the 4K BPE stage
                # runners: decode prompt, current assistant body, then the terminal EOS marker.
                text=example.prompt + example.body + BPE_EOS,
                template_sha256=frozenset(templates),
            )
            for example in examples
        ]

    from localagent.data.render import history_text

    return [
        _TrainingUnit(
            unit_id=unit_prefix,
            # Historical v1 behavior is retained so an already frozen honest manifest reproduces
            # byte-for-byte.  It is labeled as legacy rather than relabeled as a full prompt audit.
            text=history_text(messages),
            template_sha256=frozenset(templates),
        )
    ]


def _corpus_training_unit(raw: bytes, *, unit_prefix: str) -> _TrainingUnit:
    record = _strict_json_loads(raw, label=unit_prefix)
    if not isinstance(record, dict):
        raise TypeError(f"{unit_prefix} must be a JSON object")
    text_fields = [
        (key, record[key])
        for key in ("text", "content", "code")
        if isinstance(record.get(key), str)
    ]
    if not text_fields:
        raise ValueError(f"{unit_prefix} needs a text/content/code string")
    if len(text_fields) != 1:
        raise ValueError(
            f"{unit_prefix} has ambiguous text-bearing fields "
            f"{[key for key, _ in text_fields]}; exactly one is required"
        )
    _, text = text_fields[0]
    return _TrainingUnit(unit_id=unit_prefix, text=text, template_sha256=frozenset())


def _iter_training_units(
    path: Path,
    *,
    artifact_format: str,
    expected_records: int,
    max_record_bytes: int,
    artifact_name: str,
    raw_state: dict[str, Any],
    schema_version: int,
    conversation_prompt_contract: str | None,
) -> Iterator[_TrainingUnit]:
    observed_records = 0
    if artifact_format == "text":
        with path.open("rb") as handle:
            payload = handle.read(max_record_bytes + 1)
        raw_state["digest"].update(payload)
        raw_state["bytes"] += len(payload)
        if len(payload) > max_record_bytes:
            raise ValueError(f"{artifact_name} text exceeds max_record_bytes")
        observed_records = 1
        yield _TrainingUnit(
            unit_id=f"{artifact_name}:1",
            text=payload.decode("utf-8", errors="strict"),
            template_sha256=frozenset(),
        )
    else:
        with path.open("rb") as handle:
            line_number = 0
            while True:
                raw = handle.readline(max_record_bytes + 1)
                if not raw:
                    break
                raw_state["digest"].update(raw)
                raw_state["bytes"] += len(raw)
                line_number += 1
                if len(raw) > max_record_bytes:
                    raise ValueError(f"{artifact_name}:{line_number} exceeds max_record_bytes")
                if not raw.strip():
                    continue
                observed_records += 1
                unit_prefix = f"{artifact_name}:{line_number}"
                if artifact_format == "conversation_jsonl":
                    yield from _conversation_training_units(
                        raw,
                        unit_prefix=unit_prefix,
                        schema_version=schema_version,
                        conversation_prompt_contract=conversation_prompt_contract,
                    )
                else:
                    yield _corpus_training_unit(raw, unit_prefix=unit_prefix)
    if observed_records != expected_records:
        raise ValueError(
            f"{artifact_name} record-count mismatch: expected {expected_records}, "
            f"got {observed_records}"
        )


def _screen_chunk(
    units: Sequence[_TrainingUnit],
    *,
    prompts: Sequence[str],
    parameters: Mapping[str, Any],
) -> tuple[list[str], int]:
    documents = [
        CorpusDocument(
            text=unicodedata.normalize("NFKC", unit.text),
            source="training_audit",
            doc_id=unit.unit_id,
        )
        for unit in units
    ]
    retained, audit = screen_evaluation_contamination(
        documents,
        [unicodedata.normalize("NFKC", prompt) for prompt in prompts],
        shingle_size=parameters["shingle_size"],
        min_shingles=parameters["min_shingles"],
        min_coverage=parameters["min_coverage"],
        anchors_per_entry=parameters["anchors_per_entry"],
        max_denylist_shingles=parameters["max_denylist_shingles"],
    )
    retained_ids = {document.doc_id for document in retained}
    removed = sorted(
        document.doc_id for document in documents if document.doc_id not in retained_ids
    )
    return removed, int(audit["candidate_checks"])


def _audit_training_artifact(
    path: Path,
    *,
    declaration: Mapping[str, Any],
    prompts: Sequence[str],
    eval_templates: frozenset[str],
    parameters: Mapping[str, Any],
    max_record_bytes: int,
    name: str,
    expected_identity: Mapping[str, Any],
    schema_version: int,
) -> dict[str, Any]:
    expected_records = _positive_int(declaration.get("records"), name=f"{name}.records")
    artifact_format = declaration.get("format")
    if artifact_format not in _FORMATS:
        raise ValueError(f"{name}.format must be one of {sorted(_FORMATS)}")
    conversation_prompt_contract = declaration.get("conversation_prompt_contract")
    if artifact_format != "conversation_jsonl" and conversation_prompt_contract is not None:
        raise ValueError(
            f"{name}.conversation_prompt_contract is valid only for conversation_jsonl"
        )

    units_seen = 0
    candidate_checks = 0
    chunk: list[_TrainingUnit] = []
    chunk_chars = 0
    raw_state: dict[str, Any] = {"digest": hashlib.sha256(), "bytes": 0}
    for unit in _iter_training_units(
        path,
        artifact_format=artifact_format,
        expected_records=expected_records,
        max_record_bytes=max_record_bytes,
        artifact_name=name,
        raw_state=raw_state,
        schema_version=schema_version,
        conversation_prompt_contract=conversation_prompt_contract,
    ):
        units_seen += 1
        if unit.template_sha256 & eval_templates:
            raise ValueError(
                "fresh evaluation slice overlaps declared training artifacts: "
                "prompt_or_shingle_units=0, derived_action_template_units=1"
            )
        if chunk and chunk_chars + len(unit.text) > _AUDIT_CHUNK_CHARS:
            removed, checks = _screen_chunk(
                chunk,
                prompts=prompts,
                parameters=parameters,
            )
            if removed:
                raise ValueError(
                    "fresh evaluation slice overlaps declared training artifacts: "
                    "prompt_or_shingle_units=1, derived_action_template_units=0"
                )
            candidate_checks += checks
            chunk.clear()
            chunk_chars = 0
        chunk.append(unit)
        chunk_chars += len(unit.text)
        if len(chunk) == _AUDIT_CHUNK_ROWS:
            removed, checks = _screen_chunk(
                chunk,
                prompts=prompts,
                parameters=parameters,
            )
            if removed:
                raise ValueError(
                    "fresh evaluation slice overlaps declared training artifacts: "
                    "prompt_or_shingle_units=1, derived_action_template_units=0"
                )
            candidate_checks += checks
            chunk.clear()
            chunk_chars = 0
    if chunk:
        removed, checks = _screen_chunk(
            chunk,
            prompts=prompts,
            parameters=parameters,
        )
        if removed:
            raise ValueError(
                "fresh evaluation slice overlaps declared training artifacts: "
                "prompt_or_shingle_units=1, derived_action_template_units=0"
            )
        candidate_checks += checks

    observed_sha256 = raw_state["digest"].hexdigest()
    if (
        raw_state["bytes"] != expected_identity["bytes"]
        or observed_sha256 != expected_identity["sha256"]
    ):
        raise ValueError(f"{name} parsed bytes disagree with its declared identity")

    empty_fingerprint = _sha256(b"")
    result = {
        "stage": declaration["stage"],
        "name": declaration["name"],
        "format": artifact_format,
        "records": expected_records,
        "audited_units": units_seen,
        "prompt_or_shingle_overlap_units": 0,
        "prompt_or_shingle_overlap_unit_ids_sha256": empty_fingerprint,
        "derived_action_template_overlap_units": 0,
        "derived_action_template_overlap_unit_ids_sha256": empty_fingerprint,
        "candidate_checks": candidate_checks,
        "parsed_bytes": raw_state["bytes"],
        "parsed_sha256": observed_sha256,
    }
    if schema_version == HARDENED_SCHEMA_VERSION:
        result["rendered_training_text_contract"] = (
            "openai_full_catalog_v1: assistant_training_example.prompt + body + terminal EOS"
            if artifact_format == "conversation_jsonl"
            else "literal source text"
        )
        if conversation_prompt_contract is not None:
            result["conversation_prompt_contract"] = conversation_prompt_contract
    return result


def _decontamination_parameters(raw: Any) -> dict[str, Any]:
    raw = _exact_mapping(raw, _DECONTAMINATION_KEYS, label="decontamination")
    shingle_size = _positive_int(raw.get("shingle_size"), name="decontamination.shingle_size")
    min_shingles = _positive_int(raw.get("min_shingles"), name="decontamination.min_shingles")
    anchors = _positive_int(raw.get("anchors_per_entry"), name="decontamination.anchors_per_entry")
    max_shingles = _positive_int(
        raw.get("max_denylist_shingles"),
        name="decontamination.max_denylist_shingles",
    )
    if max_shingles < min_shingles:
        raise ValueError("decontamination.max_denylist_shingles must be >= min_shingles")
    return {
        "method": "unicode_nfkc_then_normalized_exact_short_or_anchor_shingle_containment",
        "shingle_size": shingle_size,
        "min_shingles": min_shingles,
        "min_coverage": _finite_fraction(
            raw.get("min_coverage"), name="decontamination.min_coverage"
        ),
        "anchors_per_entry": anchors,
        "max_denylist_shingles": max_shingles,
        "chunk_rows": _AUDIT_CHUNK_ROWS,
        "chunk_chars": _AUDIT_CHUNK_CHARS,
        "exhaustive": False,
    }


def _validate_lineage_artifacts(
    declarations: Any,
    *,
    base: Path,
    max_bytes: int,
    training_identities: Sequence[Mapping[str, Any]],
    protected_paths: set[Path],
) -> tuple[list[dict[str, Any]], set[Path]]:
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("lineage_artifacts must be a non-empty array")
    by_stage: dict[str, set[str]] = defaultdict(set)
    for identity in training_identities:
        by_stage[str(identity["stage"])].add(str(identity["sha256"]))

    observed_stages: set[str] = set()
    observed_names: set[str] = set()
    resolved_paths: set[Path] = set()
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(declarations):
        label = f"lineage_artifacts[{index}]"
        declaration = _exact_mapping(value, _LINEAGE_DECLARATION_KEYS, label=label)
        stage = declaration["stage"]
        if stage not in _HARDENED_STAGES:
            raise ValueError(f"{label}.stage must be one of {sorted(_HARDENED_STAGES)}")
        observed_stages.add(stage)
        name = _nonempty_string(declaration["name"], name=f"{label}.name")
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"{label}.name contains unsupported characters")
        if name in observed_names:
            raise ValueError(f"duplicate lineage artifact name {name!r}")
        observed_names.add(name)
        path, identity = _declared_artifact(
            declaration,
            base=base,
            name=label,
            max_bytes=max_bytes,
        )
        resolved = path.resolve()
        if resolved in protected_paths or resolved in resolved_paths:
            raise ValueError(f"{label} reuses another contract/source/training/lineage path")
        resolved_paths.add(resolved)
        payload = _read_regular_bytes(
            path,
            label=label,
            max_bytes=max_bytes,
        )
        if len(payload) != identity["bytes"] or _sha256(payload) != identity["sha256"]:
            raise ValueError(f"{label} changed before its verified bytes were parsed")
        export = _exact_mapping(
            _strict_json_loads(payload, label=label),
            _LINEAGE_EXPORT_KEYS,
            label=f"{label} export",
        )
        if export["kind"] != TRAINING_LINEAGE_KIND or export["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"{label} must be {TRAINING_LINEAGE_KIND!r} schema_version {SCHEMA_VERSION}"
            )
        if export["stage"] != stage:
            raise ValueError(f"{label} export stage disagrees with its declaration")
        checkpoint_sha256 = _sha256_string(
            export["checkpoint_sha256"],
            name=f"{label}.checkpoint_sha256",
        )
        lineage = _mapping_with_optional_keys(
            export["lineage"],
            _STAGE_LINEAGE_REQUIRED_KEYS,
            _STAGE_LINEAGE_OPTIONAL_KEYS,
            label=f"{label}.lineage",
        )
        if lineage["stage"] != stage:
            raise ValueError(f"{label}.lineage.stage disagrees with its declaration")
        if isinstance(lineage["version"], bool) or lineage["version"] != SCHEMA_VERSION:
            raise ValueError(
                f"{label}.lineage.version must be the supported version {SCHEMA_VERSION}"
            )
        for key in (
            "config_sha256",
            "model_config_sha256",
            "data_sha256",
            "tokenizer_sha256",
        ):
            _sha256_string(lineage[key], name=f"{label}.lineage.{key}")
        if "parent_checkpoint_sha256" in lineage:
            _sha256_string(
                lineage["parent_checkpoint_sha256"],
                name=f"{label}.lineage.parent_checkpoint_sha256",
            )
        git = _exact_mapping(
            lineage["git"],
            _STAGE_GIT_KEYS,
            label=f"{label}.lineage.git",
        )
        commit = git["commit"]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError(f"{label}.lineage.git.commit must be a lowercase 40-hex commit")
        _sha256_string(
            git["repository_sha256"],
            name=f"{label}.lineage.git.repository_sha256",
        )
        _sha256_string(
            git["worktree_sha256"],
            name=f"{label}.lineage.git.worktree_sha256",
        )
        if not isinstance(git["dirty"], bool):
            raise TypeError(f"{label}.lineage.git.dirty must be boolean")
        training_hashes = export["training_artifact_sha256"]
        if (
            not isinstance(training_hashes, list)
            or any(
                not isinstance(item, str) or _SHA256.fullmatch(item) is None
                for item in training_hashes
            )
            or len(set(training_hashes)) != len(training_hashes)
        ):
            raise ValueError(
                f"{label}.training_artifact_sha256 must contain unique lowercase SHA-256 values"
            )
        if set(training_hashes) != by_stage[stage]:
            raise ValueError(
                f"{label}.training_artifact_sha256 does not exactly cover declared {stage} inputs"
            )
        prompt_contract = export["conversation_prompt_contract"]
        if stage == "pretrain":
            if prompt_contract is not None:
                raise ValueError(f"{label}.conversation_prompt_contract must be null for pretrain")
        elif prompt_contract != OPENAI_FULL_CATALOG_V1:
            raise ValueError(
                f"{label}.conversation_prompt_contract must be {OPENAI_FULL_CATALOG_V1!r}"
            )
        normalized.append(
            {
                **identity,
                "stage": stage,
                "name": name,
                "checkpoint_sha256": checkpoint_sha256,
                "lineage_sha256": _sha256(_canonical_bytes(lineage)),
                "training_artifact_sha256": training_hashes,
                "conversation_prompt_contract": prompt_contract,
                **(
                    {"parent_checkpoint_sha256": lineage["parent_checkpoint_sha256"]}
                    if "parent_checkpoint_sha256" in lineage
                    else {}
                ),
            }
        )
    if observed_stages != set(_HARDENED_STAGES):
        raise ValueError(
            "lineage_artifacts must cover pretrain, midtrain, sft, and rl; "
            f"observed={sorted(observed_stages)}"
        )
    checkpoints_by_stage: dict[str, set[str]] = defaultdict(set)
    for item in normalized:
        checkpoints_by_stage[item["stage"]].add(item["checkpoint_sha256"])
    for item in normalized:
        stage = item["stage"]
        parent = item.get("parent_checkpoint_sha256")
        if stage == "pretrain":
            if parent is not None:
                raise ValueError("pretrain lineage must not declare parent_checkpoint_sha256")
            continue
        if parent is None:
            raise ValueError(f"{stage} lineage must declare parent_checkpoint_sha256")
        parent_stage = _PARENT_STAGE[stage]
        if parent not in checkpoints_by_stage[parent_stage]:
            raise ValueError(
                f"{stage} parent checkpoint is absent from frozen {parent_stage} lineage artifacts"
            )
    return normalized, resolved_paths


def _case_record(case: _ExternalCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "task_cluster_id": case.task_cluster_id,
        "template_id": case.template_id,
        "derived_template_sha256": case.derived_template_sha256,
        "source_identity": {
            "source_case_id_sha256": _sha256(case.source_case_id.encode("utf-8")),
            "source_task_cluster_id_sha256": _sha256(case.source_cluster_id.encode("utf-8")),
            "source_template_id_sha256": _sha256(case.source_template_id.encode("utf-8")),
            "source_index": case.source_index,
        },
        "family": case.family,
        "prompt": case.prompt,
        "tools": case.tools,
        "expected_calls": case.expected_calls,
        "metadata": case.metadata,
    }


def _existing_file_matches(path: Path, payload: bytes) -> bool:
    try:
        observed_bytes, observed_sha256 = _file_sha256(
            path,
            label="existing frozen output",
            max_bytes=len(payload),
        )
    except (OSError, TypeError, ValueError, RuntimeError):
        return False
    return observed_bytes == len(payload) and observed_sha256 == _sha256(payload)


def _assert_existing_or_absent(path: Path, payload: bytes) -> None:
    if path.exists() and not _existing_file_matches(path, payload):
        raise RuntimeError(f"refusing to overwrite drifted frozen artifact: {path}")


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        if not _existing_file_matches(path, payload):
            raise RuntimeError(f"refusing to overwrite drifted frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link atomically publishes the complete temporary file while
            # refusing to replace a destination created after the preflight check.
            os.link(temporary, path)
        except FileExistsError:
            if not _existing_file_matches(path, payload):
                raise RuntimeError(f"refusing to overwrite concurrently created artifact: {path}")
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    if not _existing_file_matches(path, payload):
        raise RuntimeError(f"published artifact failed byte verification: {path}")


def freeze_external_action_slice(
    contract_path: str | Path,
    *,
    slice_path: str | Path,
    denylist_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify, audit, and publish a deterministic external action slice.

    Existing outputs are idempotently accepted only when every byte matches the recomputed
    artifacts.  Any prompt/shingle or derived action-template overlap fails before publication.
    """

    contract_file = Path(contract_path)
    contract_payload = _read_regular_bytes(
        contract_file,
        label="contract",
        max_bytes=_MAX_CONTRACT_BYTES,
    )
    decoded_contract = _strict_json_loads(contract_payload, label=str(contract_file))
    if not isinstance(decoded_contract, Mapping):
        raise TypeError("contract must be a JSON object")
    schema_version = decoded_contract.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"contract schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    contract = _exact_mapping(
        decoded_contract,
        _CONTRACT_KEYS_V2 if schema_version == HARDENED_SCHEMA_VERSION else _CONTRACT_KEYS_V1,
        label="contract",
    )
    if contract.get("kind") != CONTRACT_KIND:
        raise ValueError(f"contract must be {CONTRACT_KIND!r} schema_version {schema_version}")

    limits = _exact_mapping(contract.get("limits"), _LIMIT_KEYS, label="limits")
    max_artifact_bytes = _positive_int(
        limits.get("max_artifact_bytes", _DEFAULT_MAX_ARTIFACT_BYTES),
        name="limits.max_artifact_bytes",
    )
    max_record_bytes = _positive_int(
        limits.get("max_record_bytes", _DEFAULT_MAX_RECORD_BYTES),
        name="limits.max_record_bytes",
    )
    max_source_bytes = _positive_int(limits.get("max_source_bytes"), name="limits.max_source_bytes")
    if max_source_bytes > max_artifact_bytes:
        raise ValueError("limits.max_source_bytes must be <= limits.max_artifact_bytes")
    if max_record_bytes > max_artifact_bytes:
        raise ValueError("limits.max_record_bytes must be <= limits.max_artifact_bytes")
    max_source_cases = _positive_int(limits.get("max_source_cases"), name="limits.max_source_cases")
    if max_artifact_bytes > _MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"limits.max_artifact_bytes exceeds the hard maximum {_MAX_ARTIFACT_BYTES}"
        )
    if max_source_bytes > _MAX_SOURCE_BYTES:
        raise ValueError(f"limits.max_source_bytes exceeds the hard maximum {_MAX_SOURCE_BYTES}")
    if max_record_bytes > _MAX_RECORD_BYTES:
        raise ValueError(f"limits.max_record_bytes exceeds the hard maximum {_MAX_RECORD_BYTES}")
    if max_source_cases > _MAX_SOURCE_CASES:
        raise ValueError(f"limits.max_source_cases exceeds the hard maximum {_MAX_SOURCE_CASES}")

    source_declaration = _exact_mapping(
        contract.get("source"),
        _SOURCE_DECLARATION_KEYS,
        label="source",
    )
    benchmark = _nonempty_string(source_declaration.get("benchmark"), name="source.benchmark")
    revision = _nonempty_string(source_declaration.get("revision"), name="source.revision")
    split = _nonempty_string(source_declaration.get("split"), name="source.split")
    source_path, source_identity = _declared_artifact(
        source_declaration,
        base=contract_file.parent,
        name="source",
        max_bytes=max_source_bytes,
    )
    source_payload = _read_regular_bytes(
        source_path,
        label="source",
        max_bytes=max_source_bytes,
    )
    if (
        len(source_payload) != source_identity["bytes"]
        or _sha256(source_payload) != source_identity["sha256"]
    ):
        raise ValueError("source changed before its verified bytes were parsed")
    source_cases = _load_source_cases(
        source_payload,
        source_label=str(source_path),
        expected_benchmark=benchmark,
        expected_revision=revision,
        expected_split=split,
        max_cases=max_source_cases,
        schema_version=schema_version,
    )
    del source_payload
    source_case_count = len(source_cases)

    selection_raw = _exact_mapping(
        contract.get("selection"),
        _SELECTION_KEYS,
        label="selection",
    )
    selected, selection_audit = _select_cases(source_cases, selection_raw)
    del source_cases
    prompts = [case.prompt for case in selected]
    eval_templates = frozenset(case.derived_template_sha256 for case in selected)
    parameters = _decontamination_parameters(contract.get("decontamination"))

    raw_training = contract.get("training_artifacts")
    if not isinstance(raw_training, list) or not raw_training:
        raise ValueError("training_artifacts must be a non-empty array")
    stages: set[str] = set()
    conversation_stages: set[str] = set()
    names: set[str] = set()
    resolved_paths: set[Path] = set()
    training_identities: list[dict[str, Any]] = []
    training_audits: list[dict[str, Any]] = []
    for index, declaration in enumerate(raw_training):
        label = f"training_artifacts[{index}]"
        if not isinstance(declaration, Mapping):
            raise TypeError(f"{label} must be an object")
        artifact_format = declaration.get("format")
        if schema_version == HARDENED_SCHEMA_VERSION:
            expected_keys = (
                _TRAINING_ARTIFACT_KEYS_V2_CONVERSATION
                if artifact_format == "conversation_jsonl"
                else _TRAINING_ARTIFACT_KEYS_V2_CORPUS
            )
        else:
            expected_keys = _TRAINING_ARTIFACT_KEYS_V1
        declaration = _exact_mapping(declaration, expected_keys, label=label)
        stage = declaration.get("stage")
        required_stages = (
            _HARDENED_STAGES if schema_version == HARDENED_SCHEMA_VERSION else _LEGACY_STAGES
        )
        if stage not in required_stages:
            raise ValueError(f"{label}.stage must be one of {sorted(required_stages)}")
        if artifact_format == "conversation_jsonl":
            conversation_stages.add(stage)
        name = _nonempty_string(declaration.get("name"), name=f"{label}.name")
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"{label}.name contains unsupported characters")
        if name in names:
            raise ValueError(f"duplicate training artifact name {name!r}")
        names.add(name)
        stages.add(stage)
        path, identity = _declared_artifact(
            declaration,
            base=contract_file.parent,
            name=label,
            max_bytes=max_artifact_bytes,
        )
        resolved = path.resolve()
        if resolved in {contract_file.resolve(), source_path.resolve()}:
            raise ValueError(f"{label} must not reuse the contract or external source artifact")
        if resolved in resolved_paths:
            raise ValueError(f"duplicate resolved training artifact path: {resolved}")
        resolved_paths.add(resolved)
        normalized_declaration = {**declaration, "stage": stage, "name": name}
        audit = _audit_training_artifact(
            path,
            declaration=normalized_declaration,
            prompts=prompts,
            eval_templates=eval_templates,
            parameters=parameters,
            max_record_bytes=max_record_bytes,
            name=label,
            expected_identity=identity,
            schema_version=schema_version,
        )
        training_identities.append(
            {
                **identity,
                "stage": stage,
                "name": name,
                "format": declaration.get("format"),
                "records": declaration.get("records"),
                **(
                    {"conversation_prompt_contract": declaration["conversation_prompt_contract"]}
                    if "conversation_prompt_contract" in declaration
                    else {}
                ),
            }
        )
        training_audits.append(audit)
    if stages != set(required_stages):
        raise ValueError(
            "training_artifacts must cover "
            + (
                "pretrain, midtrain, sft, and rl; "
                if schema_version == HARDENED_SCHEMA_VERSION
                else "pretrain, midtrain, and sft; "
            )
            + f"observed={sorted(stages)}"
        )
    if schema_version == HARDENED_SCHEMA_VERSION and conversation_stages & {
        "midtrain",
        "sft",
        "rl",
    } != {"midtrain", "sft", "rl"}:
        raise ValueError(
            "hardened training_artifacts require at least one conversation_jsonl input for "
            "midtrain, sft, and rl so exact prompt-contract text can be audited"
        )

    lineage_identities: list[dict[str, Any]] = []
    lineage_paths: set[Path] = set()
    if schema_version == HARDENED_SCHEMA_VERSION:
        lineage_identities, lineage_paths = _validate_lineage_artifacts(
            contract["lineage_artifacts"],
            base=contract_file.parent,
            max_bytes=max_artifact_bytes,
            training_identities=training_identities,
            protected_paths={
                contract_file.resolve(),
                source_path.resolve(),
                *resolved_paths,
            },
        )

    prompt_overlaps = sum(audit["prompt_or_shingle_overlap_units"] for audit in training_audits)
    template_overlaps = sum(
        audit["derived_action_template_overlap_units"] for audit in training_audits
    )
    if prompt_overlaps or template_overlaps:
        raise ValueError(
            "fresh evaluation slice overlaps declared training artifacts: "
            f"prompt_or_shingle_units={prompt_overlaps}, "
            f"derived_action_template_units={template_overlaps}"
        )

    analysis = _exact_mapping(contract.get("analysis"), _ANALYSIS_KEYS, label="analysis")
    resamples = _positive_int(
        analysis.get("bootstrap_resamples"), name="analysis.bootstrap_resamples"
    )
    if resamples > _MAX_BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"analysis.bootstrap_resamples exceeds the bounded maximum {_MAX_BOOTSTRAP_RESAMPLES}"
        )
    bootstrap_seed = analysis.get("bootstrap_seed")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("analysis.bootstrap_seed must be an integer")
    margin = analysis.get("exact_action_noninferiority_margin")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise TypeError("analysis.exact_action_noninferiority_margin must be a number")
    margin = float(margin)
    if not math.isfinite(margin) or not -1.0 <= margin <= 1.0:
        raise ValueError(
            "analysis.exact_action_noninferiority_margin must be finite and in [-1, 1]"
        )

    slice_file = Path(slice_path)
    denylist_file = Path(denylist_path)
    manifest_file = Path(manifest_path)
    outputs = {slice_file.resolve(), denylist_file.resolve(), manifest_file.resolve()}
    if len(outputs) != 3:
        raise ValueError("slice_path, denylist_path, and manifest_path must be different")
    protected = {
        contract_file.resolve(),
        source_path.resolve(),
        *resolved_paths,
        *lineage_paths,
    }
    if outputs & protected:
        raise ValueError("frozen outputs must not overwrite contract, source, or training files")
    if (
        _read_regular_bytes(
            contract_file,
            label="contract",
            max_bytes=_MAX_CONTRACT_BYTES,
        )
        != contract_payload
    ):
        raise ValueError("contract changed while the freeze was being constructed")

    slice_payload = _canonical_bytes(
        {
            "kind": SLICE_KIND,
            "schema_version": schema_version,
            "benchmark": benchmark,
            "revision": revision,
            "split": split,
            "cases": [_case_record(case) for case in selected],
        }
    )
    denylist_payload = _canonical_bytes(
        {
            "kind": DENYLIST_KIND,
            "schema_version": schema_version,
            "version": schema_version,
            "name": f"{benchmark}:{revision}:{split}",
            "cases": [{"case_id": case.case_id, "prompt": case.prompt} for case in selected],
        }
    )
    contract_identity = {
        "path": contract_file.name,
        "bytes": len(contract_payload),
        "sha256": _sha256(contract_payload),
    }
    manifest_without_hash = {
        "kind": MANIFEST_KIND,
        "schema_version": schema_version,
        "status": "frozen_training_disjoint_awaiting_model_evaluation",
        "contract": contract_identity,
        "source": {
            **source_identity,
            "benchmark": benchmark,
            "revision": revision,
            "split": split,
            "source_cases": source_case_count,
        },
        "selection": selection_audit,
        "training_artifacts": training_identities,
        **(
            {"lineage_artifacts": lineage_identities}
            if schema_version == HARDENED_SCHEMA_VERSION
            else {}
        ),
        "decontamination_audit": {
            "parameters": parameters,
            "artifacts": training_audits,
            "prompt_or_shingle_overlap_units": prompt_overlaps,
            "derived_action_template_overlap_units": template_overlaps,
            "derived_template_scope": (
                "gold scalar argument token replacement plus an order-insensitive tool-name "
                "multiset; "
                "only labeled Conversation action rows can be template-screened"
            ),
            "limitations": (
                "Conservative bounded normalized/shingle screening of exactly declared artifacts; "
                "not a proof against undeclared data, semantic paraphrases, or benchmark exposure "
                "outside the hashed training inputs."
            ),
        },
        "analysis": {
            "bootstrap_resamples": resamples,
            "bootstrap_seed": bootstrap_seed,
            "exact_action_noninferiority_margin": margin,
            "estimand": (
                "candidate_minus_baseline mean case exact-action accuracy; timing repetitions "
                "averaged within case; percentile bootstrap resamples task clusters"
            ),
        },
        "outputs": {
            "slice": {
                "path": slice_file.name,
                "bytes": len(slice_payload),
                "sha256": _sha256(slice_payload),
                "cases": len(selected),
            },
            "prompt_only_denylist": {
                "path": denylist_file.name,
                "bytes": len(denylist_payload),
                "sha256": _sha256(denylist_payload),
                "cases": len(selected),
            },
        },
        "isolation": {
            "evaluation_only": True,
            "expected_calls_permitted_in_training": False,
            "future_training_input": (
                "prompt_only_denylist may be used for exclusion; slice and manifest must remain "
                "evaluation-only"
            ),
            "external_timestamp_required_before_model_outputs": True,
        },
    }
    manifest_hash = _sha256(_canonical_bytes(manifest_without_hash))
    manifest_payload = _canonical_bytes(
        {**manifest_without_hash, "manifest_self_sha256": manifest_hash}
    )

    for path, payload in (
        (slice_file, slice_payload),
        (denylist_file, denylist_payload),
        (manifest_file, manifest_payload),
    ):
        _assert_existing_or_absent(path, payload)
    for path, payload in (
        (slice_file, slice_payload),
        (denylist_file, denylist_payload),
        (manifest_file, manifest_payload),
    ):
        _publish_atomic(path, payload)
    return json.loads(manifest_payload)


def validate_frozen_slice(payload: Any) -> dict[str, Any]:
    """Strictly verify and return an already-decoded public frozen slice."""

    raw = _exact_mapping(payload, _SLICE_KEYS, label="frozen slice")
    schema_version = raw.get("schema_version")
    if raw.get("kind") != SLICE_KIND or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"frozen slice must be {SLICE_KIND!r} with schema_version in "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    _nonempty_string(raw.get("benchmark"), name="frozen slice benchmark")
    _nonempty_string(raw.get("revision"), name="frozen slice revision")
    _nonempty_string(raw.get("split"), name="frozen slice split")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("frozen slice cases must be a non-empty array")
    seen: set[str] = set()
    normalized_prompts: set[str] = set()
    observed_order: list[str] = []
    for index, case in enumerate(cases):
        label = f"cases[{index}]"
        case = _exact_mapping(case, _FROZEN_CASE_KEYS, label=label)
        case_id = _nonempty_string(case.get("case_id"), name=f"{label}.case_id")
        cluster_id = _nonempty_string(case.get("task_cluster_id"), name=f"{label}.task_cluster_id")
        template_id = _nonempty_string(case.get("template_id"), name=f"{label}.template_id")
        if (
            not case_id.startswith("extcase-")
            or _SHORT_SHA256.fullmatch(case_id.removeprefix("extcase-")) is None
        ):
            raise ValueError(f"{label}.case_id is not a content-bound external case ID")
        if (
            not cluster_id.startswith("extcluster-")
            or _SHORT_SHA256.fullmatch(cluster_id.removeprefix("extcluster-")) is None
        ):
            raise ValueError(f"{label}.task_cluster_id is invalid")
        if (
            not template_id.startswith("exttemplate-")
            or _SHORT_SHA256.fullmatch(template_id.removeprefix("exttemplate-")) is None
        ):
            raise ValueError(f"{label}.template_id is invalid")
        template_sha256 = _sha256_string(
            case.get("derived_template_sha256"),
            name=f"{label}.derived_template_sha256",
        )
        if case_id in seen:
            raise ValueError(f"duplicate frozen case_id {case_id!r}")
        seen.add(case_id)
        observed_order.append(case_id)
        source_identity = _exact_mapping(
            case.get("source_identity"),
            _SOURCE_IDENTITY_KEYS,
            label=f"{label}.source_identity",
        )
        for key in (
            "source_case_id_sha256",
            "source_task_cluster_id_sha256",
            "source_template_id_sha256",
        ):
            _sha256_string(source_identity[key], name=f"{label}.source_identity.{key}")
        _nonnegative_int(
            source_identity["source_index"],
            name=f"{label}.source_identity.source_index",
        )
        _nonempty_string(case.get("family"), name=f"{label}.family")
        prompt = _nonempty_string(case.get("prompt"), name=f"{label}.prompt")
        normalized = normalize_prompt(prompt)
        if not normalized or normalized in normalized_prompts:
            raise ValueError(f"{label}.prompt is empty or duplicated after normalization")
        normalized_prompts.add(normalized)
        raw_tools = case.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValueError(f"{label}.tools must be a non-empty array")
        tools = [
            _validate_tool(tool, name=f"{label}.tools[{tool_index}]")
            for tool_index, tool in enumerate(raw_tools)
        ]
        tools_by_name = {tool["name"]: tool for tool in tools}
        if len(tools_by_name) != len(tools):
            raise ValueError(f"{label}.tools contains duplicate names")
        render_agent_decode_prompt(
            [Message(role=Role.user, content=prompt)],
            [
                ToolSpec(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"],
                )
                for tool in tools
            ],
        )
        raw_calls = case.get("expected_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError(f"{label}.expected_calls must be a non-empty array")
        calls = [
            _validate_call(
                call,
                tools=tools_by_name,
                name=f"{label}.expected_calls[{call_index}]",
            )
            for call_index, call in enumerate(raw_calls)
        ]
        if action_template_sha256(prompt, calls) != template_sha256:
            raise ValueError(f"{label}.derived_template_sha256 does not match prompt/gold calls")
        metadata = case.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError(f"{label}.metadata must be an object")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.metadata must contain finite JSON values") from error
    if observed_order != sorted(observed_order):
        raise ValueError("frozen slice cases must be sorted by case_id")
    return raw


def load_frozen_slice(path: str | Path) -> dict[str, Any]:
    """Load and strictly verify the public frozen-slice schema."""

    source = Path(path)
    payload = _strict_json_loads(
        _read_regular_bytes(
            source,
            label="frozen slice",
            max_bytes=_MAX_SOURCE_BYTES,
        ),
        label=str(source),
    )
    return validate_frozen_slice(payload)
