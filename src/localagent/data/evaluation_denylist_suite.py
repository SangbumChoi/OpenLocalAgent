"""Freeze adapter-produced benchmark prompts for pretraining decontamination.

This module is deliberately separate from fresh labeled evaluation.  It accepts only prompt-only
adapter rows, rejects label-like fields recursively, verifies every input by byte size and SHA-256,
and publishes a canonical JSONL denylist plus a portable self-hashed provenance manifest.

The resulting JSONL is accepted by :func:`localagent.data.pretrain_corpus.read_evaluation_denylist`.
It is evidence about the inputs used for corpus screening only.  It is not a labeled evaluation
slice and cannot support benchmark-score or model-quality claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

CONTRACT_KIND = "localagent_evaluation_denylist_suite_contract"
MANIFEST_KIND = "localagent_evaluation_denylist_suite_provenance"
SCHEMA_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUITE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

_MAX_CONTRACT_BYTES = 1024 * 1024
_MAX_BENCHMARK_PLAN_BYTES = 1024 * 1024
_MAX_SOURCE_ARTIFACTS = 32
_MAX_ADAPTER_PROVENANCE_ARTIFACTS = 32
_MAX_LICENSE_EVIDENCE_ARTIFACTS = 32
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_ADAPTER_PROVENANCE_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_ADAPTER_PROVENANCE_BYTES = 64 * 1024 * 1024
_MAX_LICENSE_EVIDENCE_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_LICENSE_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 250_000
# A 512 KiB UTF-8 prompt can expand to just over 3 MiB when every character requires a six-byte
# JSON escape. Keep the serialized JSONL cap above that worst case while retaining a hard bound.
_MAX_RECORD_BYTES = 4 * 1024 * 1024
_MAX_PROMPT_BYTES = 512 * 1024
_MAX_SOURCE_CASE_ID_BYTES = 4096
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024

_CONTRACT_REQUIRED_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "suite",
        "benchmark_plan",
        "sources",
        "adapter_provenance",
        "license_evidence",
        "limits",
    }
)
_CONTRACT_OPTIONAL_KEYS = frozenset({"raw_artifacts"})
_SUITE_KEYS = frozenset(
    {"name", "benchmark", "revision", "split", "adapter"}
)
_ADAPTER_KEYS = frozenset({"name", "version"})
_SOURCE_KEYS = frozenset({"name", "path", "bytes", "sha256", "records"})
_EVIDENCE_KEYS = frozenset({"name", "path", "bytes", "sha256"})
_RAW_ARTIFACT_KEYS = frozenset({"name", "path", "bytes", "sha256", "role"})
_LIMIT_KEYS = frozenset(
    {
        "max_source_bytes",
        "max_benchmark_plan_bytes",
        "max_adapter_provenance_bytes",
        "max_license_evidence_bytes",
        "max_rows",
        "max_record_bytes",
    }
)
_ROW_KEYS = frozenset({"source_case_id", "prompt"})
_MANIFEST_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "status",
        "suite",
        "benchmark_plan",
        "contract",
        "sources",
        "adapter_provenance",
        "license_evidence",
        "raw_artifacts",
        "limits",
        "deduplication_audit",
        "output",
        "isolation",
        "manifest_self_sha256",
    }
)
_MANIFEST_CONTRACT_KEYS = frozenset({"path", "bytes", "sha256"})
_MANIFEST_OUTPUT_KEYS = frozenset(
    {"path", "format", "bytes", "sha256", "records", "fields"}
)
_FREEZE_BINDING_KEYS = frozenset(
    {
        "adapter",
        "benchmark",
        "mode",
        "revision",
        "split",
        "prompt_only",
        "contains_current_step_labels",
        "output",
    }
)
_MIND2WEB_FREEZE_BINDING_KEYS = (
    _FREEZE_BINDING_KEYS | {"adapter_implementation", "ranker"}
)
_FREEZE_BINDING_OUTPUT_KEYS = frozenset({"bytes", "sha256", "records"})
_MIND2WEB_RANKER_POLICY_KEYS = frozenset(
    {
        "adapter_implementation",
        "adapter_version",
        "budget",
        "config",
        "implementation",
        "input_projection",
        "ranker_version",
        "runtime",
    }
)
_MIND2WEB_RANKER_CONFIG_POLICY_KEYS = frozenset(
    {"bytes", "config_self_sha256", "file", "sha256"}
)
_MIND2WEB_IMPLEMENTATION_POLICY_KEYS = frozenset(
    {"bytes", "module", "path", "sha256"}
)
_MIND2WEB_RANKER_BUDGET_POLICY_KEYS = frozenset(
    {
        "assistant_marker_bytes",
        "generation_reserve_tokens_including_eos",
        "max_framed_prompt_bytes",
        "max_unframed_prompt_bytes",
        "minimum_dom_bytes",
        "model_max_seq_len",
        "user_marker_bytes",
    }
)
_MIND2WEB_INPUT_PROJECTION_KEYS = frozenset({"allowed", "forbidden"})
_MIND2WEB_RUNTIME_POLICY_KEYS = frozenset(
    {
        "html_parser",
        "python_implementation",
        "python_version",
        "unicode_version",
    }
)
_MIND2WEB_ATTESTATION_KEYS = frozenset(
    {
        "archive",
        "archive_format",
        "kind",
        "members",
        "members_sha256",
        "schema_version",
        "tasks_by_split",
        "total_tasks",
        "total_uncompressed_bytes",
    }
)
_MIND2WEB_ARCHIVE_KEYS = frozenset({"bytes", "name", "sha256"})
_MIND2WEB_ARCHIVE_FORMAT_KEYS = frozenset(
    {"compression", "encryption", "members"}
)
_MIND2WEB_MEMBER_KEYS = frozenset(
    {
        "bytes",
        "compressed_bytes",
        "crc32",
        "member",
        "rows",
        "sha256",
        "split",
        "tasks",
    }
)
_MIND2WEB_SOURCE_KEYS = frozenset(
    {
        "archive_member",
        "bytes",
        "name",
        "rows",
        "sha256",
        "split",
        "tasks",
    }
)
_BFCL_SOURCE_KEYS = frozenset({"bytes", "category", "file", "rows", "sha256"})
_BFCL_SELECTION_KEYS = frozenset(
    {
        "caller_declared_categories",
        "function_spec_prompt_rows",
        "ordering",
        "question_prompt_rows",
        "source_rows",
    }
)
_WEBLINX_CHAT_SOURCE_KEYS = frozenset(
    {"bytes", "compression", "name", "sha256"}
)
_WEBLINX_SPLITS_SOURCE_KEYS = frozenset({"bytes", "name", "sha256"})
_WEBLINX_LABEL_ISOLATION_KEYS = frozenset(
    {
        "current_action_emitted",
        "expected_calls_emitted",
        "labels_emitted",
        "scores_emitted",
    }
)
_WEBLINX_PRIVACY_KEYS = frozenset(
    {
        "accepted_demos",
        "contains_private_heldout_prompts",
        "excluded_demo_id_sha256",
        "excluded_demo_ids_sha256",
        "excluded_demos",
        "excluded_rows",
        "filter_version",
        "reason_counts",
        "redistribution_authorized",
        "scanned_demos",
    }
)
_BROWSERGYM_CAPTURE_KEYS = frozenset({"bytes", "file", "rows", "sha256"})
_BROWSERGYM_CAPTURE_POLICY_KEYS = frozenset(
    {"bytes", "file", "requirement", "sha256", "status"}
)
_BROWSERGYM_CAPTURE_RECEIPT_KEYS = frozenset(
    {
        "bytes",
        "file",
        "kind",
        "producer",
        "receipt_self_sha256",
        "schema_version",
        "sha256",
    }
)
_BROWSERGYM_CAPTURE_RECEIPT_POLICY_KEYS = (
    _BROWSERGYM_CAPTURE_RECEIPT_KEYS | {"status"}
)
_BROWSERGYM_SOURCE_PIN_KEYS = frozenset(
    {"browsergym_revision", "browsergym_version", "miniwob_revision"}
)
_BROWSERGYM_RUNTIME_PIN_KEYS = frozenset(
    {
        "action_set",
        "architecture",
        "browser_executable",
        "browser_installation",
        "chromium_revision",
        "chromium_version",
        "device_scale_factor",
        "environment_manifest",
        "headless",
        "locale",
        "max_steps",
        "observation_mode",
        "os",
        "playwright_version",
        "python_version",
        "playwright_operation_timeout_seconds",
        "timezone_id",
        "viewport",
    }
)
_BROWSERGYM_PLAN_KEYS = frozenset(
    {
        "episode_rows",
        "fixed_seeds",
        "grouping_sha256",
        "localagent_policy_exclusions",
        "similarity_group_count",
        "similarity_groups",
        "splits",
        "task_groups",
        "task_variants",
    }
)
_KNOWN_SUITE_IDENTITIES: Mapping[str, Mapping[str, str]] = {
    "bfcl": {
        "adapter": "bfcl-v4-prompt-rows-v1",
        "audit_kind": "localagent_bfcl_v4_prompt_export_audit",
        "benchmark": "bfcl-v4",
        "name": "bfcl",
    },
    "browsergym": {
        "adapter": "browsergym-miniwob-reset-capture-prompt-rows-v1",
        "audit_kind": "localagent_browsergym_miniwob_prompt_export_audit",
        "benchmark": "browsergym-miniwob",
        "name": "browsergym",
    },
    "mind2web": {
        "adapter": "mind2web-private-prompt-rows-v2",
        "audit_kind": "localagent_mind2web_prompt_adapter_audit",
        "benchmark": "mind2web",
        "name": "mind2web",
    },
    "weblinx": {
        "adapter": "weblinx-private-prompt-rows-v1",
        "audit_kind": "localagent_weblinx_prompt_adapter_audit",
        "benchmark": "weblinx-chat-v1.0",
        "name": "weblinx",
    },
}
_CRC32 = re.compile(r"[0-9a-f]{8}")
_BROWSERGYM_RAW_ARTIFACT_ROLES = frozenset(
    {"browsergym_capture", "browsergym_receipt"}
)
_BFCL_RAW_SOURCE_ROLES: Mapping[str, str] = {
    "bfcl_source_multiple": "multiple",
    "bfcl_source_parallel": "parallel",
    "bfcl_source_parallel_multiple": "parallel_multiple",
    "bfcl_source_simple_python": "simple_python",
}
_BFCL_RAW_ARTIFACT_ROLES = frozenset(
    {"bfcl_source_manifest", *_BFCL_RAW_SOURCE_ROLES}
)
_WEBLINX_RAW_ARTIFACT_ROLES = frozenset(
    {"weblinx_chat_source", "weblinx_splits_source"}
)
_MIND2WEB_RAW_ARTIFACT_ROLES = frozenset(
    {"mind2web_archive_source", "mind2web_ranker_config"}
)
_RAW_ARTIFACT_HARD_CAP = max(
    len(_BFCL_RAW_ARTIFACT_ROLES),
    len(_BROWSERGYM_RAW_ARTIFACT_ROLES),
    len(_MIND2WEB_RAW_ARTIFACT_ROLES),
    len(_WEBLINX_RAW_ARTIFACT_ROLES),
)

_MUTABLE_REVISION_ALIASES = frozenset(
    {"current", "head", "latest", "main", "master", "tip", "trunk"}
)
_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "action",
        "actions",
        "answer",
        "answers",
        "completion",
        "completions",
        "expected",
        "expectation",
        "gold",
        "label",
        "labels",
        "output",
        "outputs",
        "reference",
        "references",
        "response",
        "responses",
        "solution",
        "solutions",
        "target",
        "targets",
        "toolcall",
        "toolcalls",
    }
)
_FORBIDDEN_COMPOUND_FIELDS = frozenset(
    {
        "ground_truth",
        "reference_answer",
        "tool_call",
        "tool_calls",
    }
)


@dataclass(frozen=True)
class _Artifact:
    """One verified contract artifact declaration."""

    name: str
    path: Path
    bytes: int
    sha256: str
    records: int | None = None


@dataclass(frozen=True)
class _RawArtifact:
    """One immutable raw producer artifact required to reproduce an adapter output."""

    role: str
    artifact: _Artifact


@dataclass(frozen=True)
class _PromptRow:
    """One validated source row reduced to prompt-only public material."""

    source_case_id_sha256: str
    prompt: str
    normalized_prompt: str


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("benchmark plan contains an unhashable mapping key") from error
        if duplicate:
            raise ValueError(f"benchmark plan contains duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality coercions."""

    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (TypeError, ValueError):
        return False


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read at most ``max_bytes`` from a regular non-symlink file.

    The bounded read is intentional even after a size preflight: another process can replace or
    grow a path between ``stat`` and ``open``.  Reading one sentinel byte makes that race fail
    closed without allocating an attacker-controlled file.
    """

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or is not a regular non-symlink file: {path}")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    return payload


def _matches_payload(path: Path, payload: bytes) -> bool:
    """Compare an existing artifact without an unbounded ``read_bytes`` call."""

    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != len(payload)
    ):
        return False
    offset = 0
    with path.open("rb") as handle:
        while offset < len(payload):
            chunk = handle.read(min(1024 * 1024, len(payload) - offset))
            if not chunk or chunk != payload[offset : offset + len(chunk)]:
                return False
            offset += len(chunk)
        return handle.read(1) == b""


def _portable_path(path: Path, *, relative_to: Path) -> str:
    """Return a location-independent POSIX path relative to a manifest directory."""

    return Path(
        os.path.relpath(path.resolve(), start=relative_to.resolve())
    ).as_posix()


def _joined_fingerprint(values: Sequence[str]) -> str:
    if not values:
        return _sha256(b"")
    return _sha256(("\n".join(values) + "\n").encode("utf-8"))


def _strict_json_loads(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    result = value.strip()
    try:
        result.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must contain valid Unicode scalar values") from error
    return result


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_finite_number(value: Any, *, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return value


def _safe_posix_member_name(value: Any, *, label: str) -> str:
    name = _nonempty_string(value, label=label)
    member = PurePosixPath(name)
    if (
        name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or member.as_posix() != name
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise ValueError(f"{label} must be a safe canonical POSIX member path")
    return name


def _normalized_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def _field_tokens(key: str) -> tuple[str, set[str]]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", snake).strip("_").casefold()
    return normalized, {token for token in normalized.split("_") if token}


def _reject_label_fields(value: Any, *, label: str) -> None:
    """Reject label/gold/action/output-like keys at every nesting depth."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized, tokens = _field_tokens(str(key))
            if (
                normalized in _FORBIDDEN_COMPOUND_FIELDS
                or tokens & _FORBIDDEN_FIELD_TOKENS
            ):
                raise ValueError(f"{label} contains forbidden label/action field {key!r}")
            _reject_label_fields(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_label_fields(nested, label=f"{label}[{index}]")


def _source_case_id_sha256(
    source_case_id: str,
    *,
    suite: Mapping[str, Any],
) -> str:
    semantic = {
        "benchmark": suite["benchmark"],
        "revision": suite["revision"],
        "source_case_id": source_case_id,
        "split": suite["split"],
        "suite_name": suite["name"],
    }
    return _sha256(
        b"localagent-evaluation-denylist-source-id-v1\0"
        + _canonical_bytes(semantic)
    )


def _artifact_from_record(
    raw: Any,
    *,
    base: Path,
    label: str,
    source: bool,
) -> _Artifact:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    _exact_keys(raw, _SOURCE_KEYS if source else _EVIDENCE_KEYS, label=label)
    name = _nonempty_string(raw.get("name"), label=f"{label}.name")
    if _ARTIFACT_NAME.fullmatch(name) is None:
        raise ValueError(f"{label}.name contains unsupported characters")
    raw_path = _nonempty_string(raw.get("path"), label=f"{label}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"{label}.path is missing or is not a regular non-symlink file: {path}"
        )
    if source and path.suffix.casefold() not in {".jsonl", ".ndjson"}:
        raise ValueError(f"{label}.path must be a JSONL or NDJSON file")
    expected_bytes = _positive_int(raw.get("bytes"), label=f"{label}.bytes")
    digest = raw.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256")
    records = (
        _positive_int(raw.get("records"), label=f"{label}.records")
        if source
        else None
    )
    return _Artifact(
        name=name,
        path=path,
        bytes=expected_bytes,
        sha256=digest,
        records=records,
    )


def _raw_artifact_from_record(
    raw: Any,
    *,
    base: Path,
    label: str,
) -> _RawArtifact:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    _exact_keys(raw, _RAW_ARTIFACT_KEYS, label=label)
    role = _nonempty_string(raw.get("role"), label=f"{label}.role")
    artifact = _artifact_from_record(
        {key: raw[key] for key in _EVIDENCE_KEYS},
        base=base,
        label=label,
        source=False,
    )
    return _RawArtifact(role=role, artifact=artifact)


def _validated_suite(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("suite must be an object")
    _exact_keys(raw, _SUITE_KEYS, label="suite")
    name = _nonempty_string(raw.get("name"), label="suite.name")
    if _SUITE_NAME.fullmatch(name) is None:
        raise ValueError(
            "suite.name must contain only letters, digits, '.', '_', and '-'"
        )
    benchmark = _nonempty_string(raw.get("benchmark"), label="suite.benchmark")
    revision = _nonempty_string(raw.get("revision"), label="suite.revision")
    if revision.casefold() in _MUTABLE_REVISION_ALIASES:
        raise ValueError("suite.revision must name an immutable revision, not a moving alias")
    split = _nonempty_string(raw.get("split"), label="suite.split")
    raw_adapter = raw.get("adapter")
    if not isinstance(raw_adapter, Mapping):
        raise ValueError("suite.adapter must be an object")
    _exact_keys(raw_adapter, _ADAPTER_KEYS, label="suite.adapter")
    adapter_name = _nonempty_string(raw_adapter.get("name"), label="suite.adapter.name")
    adapter_version = _nonempty_string(
        raw_adapter.get("version"), label="suite.adapter.version"
    )
    if adapter_version.casefold() in _MUTABLE_REVISION_ALIASES:
        raise ValueError("suite.adapter.version must be immutable, not a moving alias")
    return {
        "name": name,
        "benchmark": benchmark,
        "revision": revision,
        "split": split,
        "adapter": {"name": adapter_name, "version": adapter_version},
    }


def _validated_limits(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ValueError("limits must be an object")
    _exact_keys(raw, _LIMIT_KEYS, label="limits")
    limits = {
        key: _positive_int(raw.get(key), label=f"limits.{key}")
        for key in sorted(_LIMIT_KEYS)
    }
    hard_maxima = {
        "max_source_bytes": _MAX_SOURCE_BYTES,
        "max_benchmark_plan_bytes": _MAX_BENCHMARK_PLAN_BYTES,
        "max_adapter_provenance_bytes": _MAX_ADAPTER_PROVENANCE_BYTES,
        "max_license_evidence_bytes": _MAX_LICENSE_EVIDENCE_BYTES,
        "max_rows": _MAX_ROWS,
        "max_record_bytes": _MAX_RECORD_BYTES,
    }
    for key, hard_maximum in hard_maxima.items():
        if limits[key] > hard_maximum:
            raise ValueError(f"limits.{key} exceeds hard maximum {hard_maximum}")
    if limits["max_record_bytes"] > limits["max_source_bytes"]:
        raise ValueError("limits.max_record_bytes must be <= limits.max_source_bytes")
    return limits


def _verified_artifact(
    artifact: _Artifact,
    *,
    max_bytes: int,
    artifact_kind: str,
) -> dict[str, Any]:
    if artifact.path.stat().st_size != artifact.bytes:
        raise ValueError(
            f"{artifact_kind} {artifact.name!r} byte identity disagrees with contract"
        )
    observed_bytes = 0
    digest = hashlib.sha256()
    with artifact.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise ValueError(
                    f"{artifact_kind} {artifact.name!r} exceeds its declared byte limit"
                )
            digest.update(chunk)
    observed_sha256 = digest.hexdigest()
    if observed_bytes != artifact.bytes or observed_sha256 != artifact.sha256:
        raise ValueError(
            f"{artifact_kind} {artifact.name!r} byte identity disagrees with contract"
        )
    return {
        "name": artifact.name,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _verified_benchmark_plan(
    artifact: _Artifact,
    *,
    max_bytes: int,
    suite: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _verified_artifact(
        artifact,
        max_bytes=max_bytes,
        artifact_kind="benchmark plan",
    )
    payload = _read_bounded_regular_file(
        artifact.path,
        max_bytes=max_bytes,
        label="benchmark plan",
    )
    if len(payload) != identity["bytes"] or _sha256(payload) != identity["sha256"]:
        raise ValueError("benchmark plan changed while it was being verified")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("benchmark plan is not valid UTF-8") from error
    try:
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)):
                raise ValueError("benchmark plan YAML aliases and anchors are forbidden")
        plan = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("benchmark plan is not valid strict YAML") from error
    if not isinstance(plan, Mapping):
        raise ValueError("benchmark plan must be a mapping")
    if (
        plan.get("kind") != "localagent_external_benchmark_plan"
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("purpose") != "pretraining_prompt_only_decontamination"
        or plan.get("forbid_gold_in_prompt_exports") is not True
    ):
        raise ValueError("benchmark plan top-level policy is incompatible")
    prompt_freeze = plan.get("prompt_freeze")
    if not isinstance(prompt_freeze, Mapping):
        raise ValueError("benchmark plan prompt_freeze must be a mapping")
    required_freeze_policy = {
        "suite_contract_kind": CONTRACT_KIND,
        "suite_provenance_kind": MANIFEST_KIND,
        "list_manifest_kind": "localagent_evaluation_denylist_manifest",
        "schema_version": SCHEMA_VERSION,
        "require_adapter_audit_binding": True,
        "require_license_evidence_binding": True,
        "require_prompt_only_isolation": True,
    }
    for key, expected in required_freeze_policy.items():
        if prompt_freeze.get(key) != expected:
            raise ValueError(f"benchmark plan prompt_freeze.{key} is incompatible")
    external_suites = prompt_freeze.get("external_manifest_suites")
    if not isinstance(external_suites, list) or suite["name"] not in external_suites:
        raise ValueError("suite is absent from benchmark plan external_manifest_suites")
    suites = plan.get("suites")
    if not isinstance(suites, Mapping):
        raise ValueError("benchmark plan suites must be a mapping")
    suite_plan = suites.get(suite["name"])
    if not isinstance(suite_plan, Mapping):
        raise ValueError("suite is absent from benchmark plan suites")
    expected_suite_fields = {
        "benchmark": suite["benchmark"],
        "revision": suite["revision"],
        "adapter": suite["adapter"]["version"],
        "prompt_freeze_split": suite["split"],
    }
    for key, expected in expected_suite_fields.items():
        if suite_plan.get(key) != expected:
            raise ValueError(
                f"benchmark plan suite {suite['name']!r} field {key!r} "
                "disagrees with the freeze contract"
            )
    return (
        {
            **identity,
            "plan_kind": plan["kind"],
            "plan_schema_version": plan["schema_version"],
            "suite_entry_sha256": _sha256(_canonical_bytes(dict(suite_plan))),
        },
        dict(suite_plan),
    )


def _validated_specialized_output(
    audit: Mapping[str, Any],
    *,
    suite: Mapping[str, Any],
    expected_kind: str,
    expected_schema_version: int,
    label: str,
    require_self_hash: bool,
) -> tuple[Mapping[str, Any], int]:
    """Validate common fields emitted by a known production adapter."""

    if (
        audit.get("kind") != expected_kind
        or audit.get("schema_version") != expected_schema_version
        or isinstance(audit.get("schema_version"), bool)
    ):
        raise ValueError(f"{label} kind/schema disagrees with the production adapter")
    expected_identity = {
        "benchmark": suite["benchmark"],
        "mode": "production",
        "revision": suite["revision"],
        "split": suite["split"],
    }
    for key, expected in expected_identity.items():
        if audit.get(key) != expected:
            raise ValueError(f"{label}.{key} disagrees with the production suite")
    if require_self_hash:
        audit_self_sha256 = audit.get("audit_self_sha256")
        if (
            not isinstance(audit_self_sha256, str)
            or _SHA256.fullmatch(audit_self_sha256) is None
        ):
            raise ValueError(f"{label}.audit_self_sha256 is required")

    output = audit.get("output")
    freeze_binding = audit.get("freeze_binding")
    if not isinstance(output, Mapping) or not isinstance(freeze_binding, Mapping):
        raise ValueError(f"{label} output/freeze binding is invalid")
    binding_output = freeze_binding.get("output")
    if not isinstance(binding_output, Mapping):
        raise ValueError(f"{label}.freeze_binding.output is invalid")
    output_rows = _positive_int(output.get("rows"), label=f"{label}.output.rows")
    if (
        output.get("bytes") != binding_output.get("bytes")
        or output.get("sha256") != binding_output.get("sha256")
        or output_rows != binding_output.get("records")
    ):
        raise ValueError(f"{label} output disagrees with freeze binding")
    return output, output_rows


def _validate_bfcl_source_attestation(
    audit: Mapping[str, Any],
    *,
    suite: Mapping[str, Any],
    suite_plan: Mapping[str, Any],
    raw_reference_audit: Mapping[str, Any] | None,
    label: str,
) -> None:
    output, output_rows = _validated_specialized_output(
        audit,
        suite=suite,
        expected_kind=_KNOWN_SUITE_IDENTITIES["bfcl"]["audit_kind"],
        expected_schema_version=1,
        label=label,
        require_self_hash=False,
    )
    if output.get("row_keys") != ["prompt", "source_case_id"]:
        raise ValueError(f"{label}.output.row_keys is not the BFCL prompt-only schema")
    if raw_reference_audit is None:
        raise ValueError("BFCL freeze requires contract-bound raw source artifacts")
    for key in (
        "freeze_binding",
        "limits",
        "output",
        "selection",
        "source_manifest",
        "sources",
    ):
        if not _canonical_equal(audit.get(key), raw_reference_audit.get(key)):
            raise ValueError(
                f"{label}.{key} disagrees with the raw-source reexport"
            )

    raw_categories = suite_plan.get("categories")
    raw_plan_sources = suite_plan.get("pinned_prompt_sources")
    if not isinstance(raw_categories, list) or not isinstance(raw_plan_sources, Mapping):
        raise ValueError("benchmark plan BFCL source policy is missing")
    categories = [
        _nonempty_string(category, label="benchmark plan BFCL category")
        for category in raw_categories
    ]
    if len(categories) != len(set(categories)) or set(categories) != set(raw_plan_sources):
        raise ValueError("benchmark plan BFCL categories and pinned sources disagree")
    expected_sources: dict[str, dict[str, Any]] = {}
    for category in categories:
        raw_source = raw_plan_sources.get(category)
        source_label = f"benchmark plan BFCL source {category!r}"
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"{source_label} must be an object")
        source_sha256 = raw_source.get("sha256")
        if (
            not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
        ):
            raise ValueError(f"{source_label}.sha256 is invalid")
        expected_sources[category] = {
            "bytes": _positive_int(raw_source.get("bytes"), label=f"{source_label}.bytes"),
            "category": category,
            "file": _nonempty_string(
                raw_source.get("file"),
                label=f"{source_label}.file",
            ),
            "rows": _positive_int(raw_source.get("rows"), label=f"{source_label}.rows"),
            "sha256": source_sha256,
        }

    raw_sources = audit.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(expected_sources):
        raise ValueError(f"{label}.sources must bind every pinned BFCL source")
    observed_sources: dict[str, dict[str, Any]] = {}
    observed_order: list[str] = []
    for index, source in enumerate(raw_sources):
        source_label = f"{label}.sources[{index}]"
        if not isinstance(source, Mapping):
            raise ValueError(f"{source_label} must be an object")
        _exact_keys(source, _BFCL_SOURCE_KEYS, label=source_label)
        category = _nonempty_string(
            source.get("category"),
            label=f"{source_label}.category",
        )
        if category in observed_sources:
            raise ValueError(f"{label}.sources repeats BFCL category {category!r}")
        source_sha256 = source.get("sha256")
        if (
            not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
        ):
            raise ValueError(f"{source_label}.sha256 is invalid")
        observed_order.append(category)
        observed_sources[category] = {
            "bytes": _positive_int(source.get("bytes"), label=f"{source_label}.bytes"),
            "category": category,
            "file": _nonempty_string(source.get("file"), label=f"{source_label}.file"),
            "rows": _positive_int(source.get("rows"), label=f"{source_label}.rows"),
            "sha256": source_sha256,
        }
    if observed_order != sorted(observed_order) or observed_sources != expected_sources:
        raise ValueError(f"{label}.sources disagree with pinned BFCL source identities")

    selection = audit.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{label}.selection must be an object")
    _exact_keys(selection, _BFCL_SELECTION_KEYS, label=f"{label}.selection")
    selected_categories = selection.get("caller_declared_categories")
    if (
        not isinstance(selected_categories, list)
        or selected_categories != observed_order
    ):
        raise ValueError(f"{label}.selection categories disagree with audited sources")
    source_rows = _positive_int(
        selection.get("source_rows"),
        label=f"{label}.selection.source_rows",
    )
    expected_input_rows = _positive_int(
        suite_plan.get("expected_input_rows"),
        label="benchmark plan BFCL expected_input_rows",
    )
    if source_rows != sum(source["rows"] for source in observed_sources.values()):
        raise ValueError(f"{label}.selection source-row accounting mismatch")
    if source_rows != expected_input_rows:
        raise ValueError(f"{label}.selection disagrees with BFCL expected input rows")
    question_rows = _positive_int(
        selection.get("question_prompt_rows"),
        label=f"{label}.selection.question_prompt_rows",
    )
    function_rows = _positive_int(
        selection.get("function_spec_prompt_rows"),
        label=f"{label}.selection.function_spec_prompt_rows",
    )
    if output_rows != question_rows + function_rows:
        raise ValueError(f"{label}.selection prompt-row accounting mismatch")


def _is_browsergym_suite(suite: Mapping[str, Any]) -> bool:
    expected = _KNOWN_SUITE_IDENTITIES["browsergym"]
    return (
        suite.get("name") == expected["name"]
        and suite.get("benchmark") == expected["benchmark"]
        and isinstance(suite.get("adapter"), Mapping)
        and suite["adapter"].get("version") == expected["adapter"]
    )


def _is_bfcl_suite(suite: Mapping[str, Any]) -> bool:
    expected = _KNOWN_SUITE_IDENTITIES["bfcl"]
    return (
        suite.get("name") == expected["name"]
        and suite.get("benchmark") == expected["benchmark"]
        and isinstance(suite.get("adapter"), Mapping)
        and suite["adapter"].get("version") == expected["adapter"]
    )


def _is_mind2web_suite(suite: Mapping[str, Any]) -> bool:
    expected = _KNOWN_SUITE_IDENTITIES["mind2web"]
    return (
        suite.get("name") == expected["name"]
        and suite.get("benchmark") == expected["benchmark"]
        and isinstance(suite.get("adapter"), Mapping)
        and suite["adapter"].get("version") == expected["adapter"]
    )


def _is_weblinx_suite(suite: Mapping[str, Any]) -> bool:
    expected = _KNOWN_SUITE_IDENTITIES["weblinx"]
    return (
        suite.get("name") == expected["name"]
        and suite.get("benchmark") == expected["benchmark"]
        and isinstance(suite.get("adapter"), Mapping)
        and suite["adapter"].get("version") == expected["adapter"]
    )


def _verified_bfcl_raw_chain(
    raw_artifacts: Sequence[_RawArtifact],
    *,
    sources: Sequence[_Artifact],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if len(raw_artifacts) != len(_BFCL_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "BFCL contract must declare exactly one source manifest and four pinned sources"
        )
    by_role: dict[str, _Artifact] = {}
    for raw in raw_artifacts:
        if raw.role not in _BFCL_RAW_ARTIFACT_ROLES:
            raise ValueError(f"unsupported BFCL raw artifact role {raw.role!r}")
        if raw.role in by_role:
            raise ValueError(f"duplicate BFCL raw artifact role {raw.role!r}")
        by_role[raw.role] = raw.artifact
    if set(by_role) != set(_BFCL_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "BFCL contract raw artifacts must include the source manifest and all four sources"
        )
    if len(sources) != 1:
        raise ValueError("BFCL contract must declare exactly one prompt source")

    from localagent.data.bfcl_prompts import BFCLPromptLimits, export_bfcl_prompt_rows

    prompt_limits = BFCLPromptLimits()
    maximum_by_role = {
        role: (
            prompt_limits.max_manifest_bytes
            if role == "bfcl_source_manifest"
            else prompt_limits.max_source_bytes
        )
        for role in _BFCL_RAW_ARTIFACT_ROLES
    }
    verified_identities = {
        role: _verified_artifact(
            artifact,
            max_bytes=maximum_by_role[role],
            artifact_kind=f"BFCL raw {role}",
        )
        for role, artifact in by_role.items()
    }
    source_manifest = by_role["bfcl_source_manifest"]
    source = sources[0]
    with tempfile.TemporaryDirectory(prefix="localagent-bfcl-freeze-") as temporary:
        temporary_root = Path(temporary)
        derived_output = temporary_root / source.path.name
        derived_audit = temporary_root / "bfcl-derived-audit.json"
        reference_audit = export_bfcl_prompt_rows(
            source_manifest.path,
            derived_output,
            derived_audit,
            limits=prompt_limits,
        )
        derived_payload = _read_bounded_regular_file(
            derived_output,
            max_bytes=_MAX_SOURCE_BYTES,
            label="rederived BFCL prompt output",
        )

    manifest_payload = _read_bounded_regular_file(
        source_manifest.path,
        max_bytes=prompt_limits.max_manifest_bytes,
        label="BFCL raw source manifest",
    )
    manifest = _strict_json_loads(manifest_payload, label="BFCL raw source manifest")
    raw_manifest_sources = manifest.get("sources") if isinstance(manifest, Mapping) else None
    if not isinstance(raw_manifest_sources, list):
        raise ValueError("BFCL raw source manifest sources are invalid")
    manifest_sources: dict[str, Mapping[str, Any]] = {}
    for raw_source in raw_manifest_sources:
        if not isinstance(raw_source, Mapping):
            raise ValueError("BFCL raw source manifest source is invalid")
        category = raw_source.get("category")
        if not isinstance(category, str) or category in manifest_sources:
            raise ValueError("BFCL raw source manifest categories are invalid")
        manifest_sources[category] = raw_source
    if set(manifest_sources) != set(_BFCL_RAW_SOURCE_ROLES.values()):
        raise ValueError("BFCL raw source manifest must bind all four production categories")
    for role, category in _BFCL_RAW_SOURCE_ROLES.items():
        raw_source = manifest_sources[category]
        raw_path = raw_source.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"BFCL raw source manifest path for {category!r} is invalid")
        manifest_source_path = Path(raw_path)
        if not manifest_source_path.is_absolute():
            manifest_source_path = source_manifest.path.parent / manifest_source_path
        artifact = by_role[role]
        if manifest_source_path.resolve() != artifact.path.resolve():
            raise ValueError(
                f"BFCL raw source manifest path for {category!r} is not the contracted artifact"
            )
        if (
            raw_source.get("bytes") != artifact.bytes
            or raw_source.get("sha256") != artifact.sha256
        ):
            raise ValueError(
                f"BFCL raw source manifest identity for {category!r} "
                "disagrees with the contract"
            )

    declared_payload = _read_bounded_regular_file(
        source.path,
        max_bytes=max(source.bytes, 1),
        label="declared BFCL prompt source",
    )
    if declared_payload != derived_payload:
        raise ValueError(
            "declared BFCL prompt source differs from the raw-source reexport"
        )
    if (
        len(derived_payload) != source.bytes
        or _sha256(derived_payload) != source.sha256
    ):
        raise ValueError("rederived BFCL prompt output identity disagrees with the contract")

    for role, artifact in by_role.items():
        if (
            _verified_artifact(
                artifact,
                max_bytes=maximum_by_role[role],
                artifact_kind=f"BFCL raw {role}",
            )
            != verified_identities[role]
        ):
            raise ValueError("BFCL raw artifacts changed during freeze")
    manifest_identities = [
        {
            **verified_identities[role],
            "role": role,
        }
        for role in sorted(verified_identities)
    ]
    return manifest_identities, reference_audit


def _verified_browsergym_raw_chain(
    raw_artifacts: Sequence[_RawArtifact],
    *,
    sources: Sequence[_Artifact],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if len(raw_artifacts) != len(_BROWSERGYM_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "BrowserGym contract must declare exactly one raw capture and one raw receipt"
        )
    by_role: dict[str, _Artifact] = {}
    for raw in raw_artifacts:
        if raw.role not in _BROWSERGYM_RAW_ARTIFACT_ROLES:
            raise ValueError(f"unsupported BrowserGym raw artifact role {raw.role!r}")
        if raw.role in by_role:
            raise ValueError(f"duplicate BrowserGym raw artifact role {raw.role!r}")
        by_role[raw.role] = raw.artifact
    if set(by_role) != set(_BROWSERGYM_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "BrowserGym contract raw artifacts must include capture and receipt"
        )
    if len(sources) != 1:
        raise ValueError("BrowserGym contract must declare exactly one prompt source")

    from localagent.data.browsergym_capture import DEFAULT_MAX_RECEIPT_BYTES
    from localagent.data.browsergym_prompts import (
        DEFAULT_MAX_CAPTURE_BYTES,
        export_browsergym_prompt_rows,
    )

    capture = by_role["browsergym_capture"]
    receipt = by_role["browsergym_receipt"]
    verified_identities = {
        "browsergym_capture": _verified_artifact(
            capture,
            max_bytes=DEFAULT_MAX_CAPTURE_BYTES,
            artifact_kind="BrowserGym raw capture",
        ),
        "browsergym_receipt": _verified_artifact(
            receipt,
            max_bytes=DEFAULT_MAX_RECEIPT_BYTES,
            artifact_kind="BrowserGym raw receipt",
        ),
    }
    source = sources[0]
    with tempfile.TemporaryDirectory(prefix="localagent-browsergym-freeze-") as temporary:
        temporary_root = Path(temporary)
        derived_output = temporary_root / source.path.name
        derived_audit = temporary_root / "browsergym-derived-audit.json"
        reference_audit = export_browsergym_prompt_rows(
            capture.path,
            derived_output,
            derived_audit,
            expected_capture_bytes=capture.bytes,
            expected_capture_sha256=capture.sha256,
            receipt_path=receipt.path,
            production=True,
        )
        derived_payload = _read_bounded_regular_file(
            derived_output,
            max_bytes=_MAX_SOURCE_BYTES,
            label="rederived BrowserGym prompt output",
        )
    declared_payload = _read_bounded_regular_file(
        source.path,
        max_bytes=max(source.bytes, 1),
        label="declared BrowserGym prompt source",
    )
    if declared_payload != derived_payload:
        raise ValueError(
            "declared BrowserGym prompt source differs from the receipt-verified capture"
        )
    if (
        len(derived_payload) != source.bytes
        or _sha256(derived_payload) != source.sha256
    ):
        raise ValueError(
            "rederived BrowserGym prompt output identity disagrees with the contract"
        )

    # Recheck both raw contract identities after the verifier and derivation consumed them.
    for role, artifact in by_role.items():
        maximum = (
            DEFAULT_MAX_CAPTURE_BYTES
            if role == "browsergym_capture"
            else DEFAULT_MAX_RECEIPT_BYTES
        )
        if (
            _verified_artifact(
                artifact,
                max_bytes=maximum,
                artifact_kind=f"BrowserGym raw {role}",
            )
            != verified_identities[role]
        ):
            raise ValueError("BrowserGym raw artifacts changed during freeze")
    manifest_identities = [
        {
            **verified_identities[role],
            "role": role,
        }
        for role in sorted(verified_identities)
    ]
    return manifest_identities, reference_audit


def _verified_weblinx_raw_chain(
    raw_artifacts: Sequence[_RawArtifact],
    *,
    sources: Sequence[_Artifact],
    suite: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if len(raw_artifacts) != len(_WEBLINX_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "WebLINX contract must declare exactly one compact chat source "
            "and one splits source"
        )
    by_role: dict[str, _Artifact] = {}
    for raw in raw_artifacts:
        if raw.role not in _WEBLINX_RAW_ARTIFACT_ROLES:
            raise ValueError(f"unsupported WebLINX raw artifact role {raw.role!r}")
        if raw.role in by_role:
            raise ValueError(f"duplicate WebLINX raw artifact role {raw.role!r}")
        by_role[raw.role] = raw.artifact
    if set(by_role) != set(_WEBLINX_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "WebLINX contract raw artifacts must include compact chat and splits"
        )
    if len(sources) != 1:
        raise ValueError("WebLINX contract must declare exactly one prompt source")

    from localagent.data.weblinx_prompts import (
        DEFAULT_MAX_CHAT_SOURCE_BYTES,
        DEFAULT_MAX_SPLITS_BYTES,
        WebLINXSource,
        export_weblinx_prompt_rows,
    )

    maximum_by_role = {
        "weblinx_chat_source": DEFAULT_MAX_CHAT_SOURCE_BYTES,
        "weblinx_splits_source": DEFAULT_MAX_SPLITS_BYTES,
    }
    verified_identities = {
        role: _verified_artifact(
            artifact,
            max_bytes=maximum_by_role[role],
            artifact_kind=f"WebLINX raw {role}",
        )
        for role, artifact in by_role.items()
    }
    chat = by_role["weblinx_chat_source"]
    splits = by_role["weblinx_splits_source"]
    source = sources[0]
    with tempfile.TemporaryDirectory(prefix="localagent-weblinx-freeze-") as temporary:
        temporary_root = Path(temporary)
        derived_output = temporary_root / source.path.name
        derived_audit = temporary_root / "weblinx-derived-audit.json"
        reference_audit = export_weblinx_prompt_rows(
            WebLINXSource(
                path=chat.path,
                bytes=chat.bytes,
                sha256=chat.sha256,
            ),
            WebLINXSource(
                path=splits.path,
                bytes=splits.bytes,
                sha256=splits.sha256,
            ),
            derived_output,
            revision=suite["revision"],
            split=suite["split"],
            audit_path=derived_audit,
        )
        derived_payload = _read_bounded_regular_file(
            derived_output,
            max_bytes=_MAX_SOURCE_BYTES,
            label="rederived WebLINX prompt output",
        )

    declared_payload = _read_bounded_regular_file(
        source.path,
        max_bytes=max(source.bytes, 1),
        label="declared WebLINX prompt source",
    )
    if declared_payload != derived_payload:
        raise ValueError(
            "declared WebLINX prompt source differs from the raw-source reexport"
        )
    if (
        len(derived_payload) != source.bytes
        or _sha256(derived_payload) != source.sha256
    ):
        raise ValueError(
            "rederived WebLINX prompt output identity disagrees with the contract"
        )

    for role, artifact in by_role.items():
        if (
            _verified_artifact(
                artifact,
                max_bytes=maximum_by_role[role],
                artifact_kind=f"WebLINX raw {role}",
            )
            != verified_identities[role]
        ):
            raise ValueError("WebLINX raw artifacts changed during freeze")
    manifest_identities = [
        {
            **verified_identities[role],
            "role": role,
        }
        for role in sorted(verified_identities)
    ]
    return manifest_identities, reference_audit


def _verified_mind2web_raw_chain(
    raw_artifacts: Sequence[_RawArtifact],
    *,
    sources: Sequence[_Artifact],
    suite: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    """Rebuild Mind2Web prompts from the contracted archive and ranker config."""

    if len(raw_artifacts) != len(_MIND2WEB_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "Mind2Web contract must declare exactly one protected archive "
            "and one ranker config"
        )
    by_role: dict[str, _Artifact] = {}
    for raw in raw_artifacts:
        if raw.role not in _MIND2WEB_RAW_ARTIFACT_ROLES:
            raise ValueError(f"unsupported Mind2Web raw artifact role {raw.role!r}")
        if raw.role in by_role:
            raise ValueError(f"duplicate Mind2Web raw artifact role {raw.role!r}")
        by_role[raw.role] = raw.artifact
    if set(by_role) != set(_MIND2WEB_RAW_ARTIFACT_ROLES):
        raise ValueError(
            "Mind2Web contract raw artifacts must include archive and ranker config"
        )
    if len(sources) != 1:
        raise ValueError("Mind2Web contract must declare exactly one prompt source")

    from localagent.data.mind2web_dom_ranker import (
        load_mind2web_dom_ranker_config,
    )
    from localagent.data.mind2web_prompts import (
        DEFAULT_MAX_ARCHIVE_BYTES,
        DEFAULT_MAX_SOURCE_BYTES,
        DEFAULT_MAX_TOTAL_SOURCE_BYTES,
        PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED,
        PRODUCTION_MIND2WEB_ARCHIVE_PASSWORD,
        PRODUCTION_MIND2WEB_MEMBERS,
        Mind2WebArchive,
        Mind2WebSource,
        export_mind2web_prompt_rows,
    )

    archive_artifact = by_role["mind2web_archive_source"]
    ranker_artifact = by_role["mind2web_ranker_config"]
    maximum_by_role = {
        "mind2web_archive_source": DEFAULT_MAX_ARCHIVE_BYTES,
        "mind2web_ranker_config": 64 * 1024,
    }
    verified_identities = {
        role: _verified_artifact(
            artifact,
            max_bytes=maximum_by_role[role],
            artifact_kind=f"Mind2Web raw {role}",
        )
        for role, artifact in by_role.items()
    }
    # Validate the contracted config before spending disk or CPU on the protected archive.
    load_mind2web_dom_ranker_config(ranker_artifact.path)

    expected_members = sorted(
        member
        for members in PRODUCTION_MIND2WEB_MEMBERS.values()
        for member in members
    )
    if not expected_members:
        raise ValueError("Mind2Web production member policy is empty")
    source = sources[0]
    with tempfile.TemporaryDirectory(prefix="localagent-mind2web-freeze-") as temporary:
        temporary_root = Path(temporary)
        extracted_root = temporary_root / "members"
        extracted_root.mkdir()
        extracted_sources: list[Mind2WebSource] = []
        total_extracted_bytes = 0
        try:
            with zipfile.ZipFile(archive_artifact.path, mode="r") as archive:
                infos = archive.infolist()
                observed_names = [info.filename for info in infos]
                if (
                    set(observed_names) != set(expected_members)
                    or len(observed_names) != len(set(observed_names))
                ):
                    raise ValueError(
                        "Mind2Web protected archive member set disagrees "
                        "with production policy"
                    )
                info_by_name = {info.filename: info for info in infos}
                password = (
                    PRODUCTION_MIND2WEB_ARCHIVE_PASSWORD
                    if PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED
                    else None
                )
                observed_basenames: set[str] = set()
                for member in expected_members:
                    info = info_by_name[member]
                    member = _safe_posix_member_name(
                        member,
                        label="Mind2Web protected archive member",
                    )
                    if info.is_dir() or info.file_size <= 0:
                        raise ValueError(
                            f"Mind2Web protected archive member is not a non-empty file: {member}"
                        )
                    if info.file_size > DEFAULT_MAX_SOURCE_BYTES:
                        raise ValueError(
                            f"Mind2Web protected archive member exceeds source cap: {member}"
                        )
                    basename = PurePosixPath(member).name
                    if basename in observed_basenames:
                        raise ValueError(
                            "Mind2Web protected archive member basenames must be unique"
                        )
                    observed_basenames.add(basename)
                    extracted_path = extracted_root / basename
                    digest = hashlib.sha256()
                    member_bytes = 0
                    with archive.open(info, mode="r", pwd=password) as source_handle:
                        with extracted_path.open("xb") as output_handle:
                            for chunk in iter(
                                lambda: source_handle.read(1024 * 1024),
                                b"",
                            ):
                                member_bytes += len(chunk)
                                total_extracted_bytes += len(chunk)
                                if member_bytes > DEFAULT_MAX_SOURCE_BYTES:
                                    raise ValueError(
                                        "Mind2Web protected archive member exceeds "
                                        f"source cap while extracting: {member}"
                                    )
                                if total_extracted_bytes > DEFAULT_MAX_TOTAL_SOURCE_BYTES:
                                    raise ValueError(
                                        "Mind2Web protected archive exceeds total "
                                        "source cap while extracting"
                                    )
                                digest.update(chunk)
                                output_handle.write(chunk)
                    if member_bytes != info.file_size:
                        raise ValueError(
                            f"Mind2Web protected archive member size changed: {member}"
                        )
                    extracted_sources.append(
                        Mind2WebSource(
                            path=extracted_path,
                            bytes=member_bytes,
                            sha256=digest.hexdigest(),
                            archive_member=member,
                        )
                    )
        except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
            raise ValueError("cannot extract contracted Mind2Web archive") from error

        derived_output = temporary_root / source.path.name
        derived_audit = temporary_root / "mind2web-derived-audit.json"
        reference_audit = export_mind2web_prompt_rows(
            extracted_sources,
            derived_output,
            revision=str(suite["revision"]),
            split=str(suite["split"]),
            archive=Mind2WebArchive(
                path=archive_artifact.path,
                bytes=archive_artifact.bytes,
                sha256=archive_artifact.sha256,
            ),
            audit_path=derived_audit,
            ranker_config_path=ranker_artifact.path,
        )
        derived_payload = _read_bounded_regular_file(
            derived_output,
            max_bytes=_MAX_SOURCE_BYTES,
            label="rederived Mind2Web prompt output",
        )

    declared_payload = _read_bounded_regular_file(
        source.path,
        max_bytes=max(source.bytes, 1),
        label="declared Mind2Web prompt source",
    )
    if declared_payload != derived_payload:
        raise ValueError(
            "declared Mind2Web prompt source differs from the raw-source reexport"
        )
    if (
        len(derived_payload) != source.bytes
        or _sha256(derived_payload) != source.sha256
    ):
        raise ValueError(
            "rederived Mind2Web prompt output identity disagrees with the contract"
        )

    for role, artifact in by_role.items():
        if (
            _verified_artifact(
                artifact,
                max_bytes=maximum_by_role[role],
                artifact_kind=f"Mind2Web raw {role}",
            )
            != verified_identities[role]
        ):
            raise ValueError("Mind2Web raw artifacts changed during freeze")
    manifest_identities = [
        {
            **verified_identities[role],
            "role": role,
        }
        for role in sorted(verified_identities)
    ]
    return manifest_identities, reference_audit


def _validate_browsergym_source_attestation(
    audit: Mapping[str, Any],
    *,
    suite: Mapping[str, Any],
    suite_plan: Mapping[str, Any],
    raw_reference_audit: Mapping[str, Any] | None,
    label: str,
) -> None:
    from localagent.data import browsergym_prompts

    output, output_rows = _validated_specialized_output(
        audit,
        suite=suite,
        expected_kind=_KNOWN_SUITE_IDENTITIES["browsergym"]["audit_kind"],
        expected_schema_version=2,
        label=label,
        require_self_hash=True,
    )
    if raw_reference_audit is None:
        raise ValueError(
            "BrowserGym freeze requires receipt-verified raw capture artifacts"
        )
    for key in (
        "capture",
        "capture_receipt",
        "freeze_binding",
        "output",
        "plan",
        "runtime_pins",
        "source_pins",
    ):
        if not _canonical_equal(audit.get(key), raw_reference_audit.get(key)):
            raise ValueError(
                f"{label}.{key} disagrees with the receipt-verified raw capture"
            )
    if output.get("row_keys") != ["prompt", "source_case_id"]:
        raise ValueError(
            f"{label}.output.row_keys is not the BrowserGym prompt-only schema"
        )

    capture_policy = suite_plan.get("prompt_capture")
    if not isinstance(capture_policy, Mapping):
        raise ValueError("benchmark plan BrowserGym prompt_capture is missing")
    _exact_keys(
        capture_policy,
        _BROWSERGYM_CAPTURE_POLICY_KEYS,
        label="benchmark plan BrowserGym prompt_capture",
    )
    capture_sha256 = capture_policy.get("sha256")
    if (
        capture_policy.get("status") != "frozen_controlled_acquisition"
        or not isinstance(capture_sha256, str)
        or _SHA256.fullmatch(capture_sha256) is None
    ):
        raise ValueError(
            "benchmark plan BrowserGym capture remains pending or lacks a frozen SHA-256"
        )
    capture_bytes = _positive_int(
        capture_policy.get("bytes"),
        label="benchmark plan BrowserGym prompt_capture.bytes",
    )
    capture_file = _nonempty_string(
        capture_policy.get("file"),
        label="benchmark plan BrowserGym prompt_capture.file",
    )
    if capture_policy.get("requirement") != "freeze_before_tokenizer_fit":
        raise ValueError("benchmark plan BrowserGym prompt_capture requirement drift")
    if (
        browsergym_prompts.PRODUCTION_CAPTURE_BYTES is None
        or browsergym_prompts.PRODUCTION_CAPTURE_SHA256 is None
    ):
        raise ValueError(
            "production BrowserGym capture constants remain pending"
        )
    if (
        capture_file != browsergym_prompts.PRODUCTION_CAPTURE_FILE
        or capture_bytes != browsergym_prompts.PRODUCTION_CAPTURE_BYTES
        or capture_sha256 != browsergym_prompts.PRODUCTION_CAPTURE_SHA256
    ):
        raise ValueError(
            "benchmark plan BrowserGym prompt_capture disagrees with "
            "production capture constants"
        )
    capture = audit.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError(f"{label}.capture must be an object")
    _exact_keys(capture, _BROWSERGYM_CAPTURE_KEYS, label=f"{label}.capture")
    if (
        capture.get("file") != capture_file
        or capture.get("bytes") != capture_bytes
        or capture.get("sha256") != capture_sha256
        or capture.get("rows") != output_rows
    ):
        raise ValueError(f"{label}.capture disagrees with the frozen BrowserGym plan")
    _nonempty_string(capture.get("file"), label=f"{label}.capture.file")

    receipt_policy = suite_plan.get("capture_receipt")
    if not isinstance(receipt_policy, Mapping):
        raise ValueError("benchmark plan BrowserGym capture_receipt is missing")
    _exact_keys(
        receipt_policy,
        _BROWSERGYM_CAPTURE_RECEIPT_POLICY_KEYS,
        label="benchmark plan BrowserGym capture_receipt",
    )
    expected_receipt = {
        key: receipt_policy[key] for key in _BROWSERGYM_CAPTURE_RECEIPT_KEYS
    }
    receipt_sha256 = expected_receipt["sha256"]
    receipt_self_sha256 = expected_receipt["receipt_self_sha256"]
    if (
        receipt_policy.get("status") != "frozen_controlled_acquisition"
        or any(value is None for value in expected_receipt.values())
        or not isinstance(receipt_sha256, str)
        or _SHA256.fullmatch(receipt_sha256) is None
        or not isinstance(receipt_self_sha256, str)
        or _SHA256.fullmatch(receipt_self_sha256) is None
    ):
        raise ValueError(
            "benchmark plan BrowserGym receipt remains pending or lacks "
            "an exact frozen identity"
        )
    _positive_int(
        expected_receipt["bytes"],
        label="benchmark plan BrowserGym capture_receipt.bytes",
    )
    _positive_int(
        expected_receipt["schema_version"],
        label="benchmark plan BrowserGym capture_receipt.schema_version",
    )
    for key in ("file", "kind", "producer"):
        _nonempty_string(
            expected_receipt[key],
            label=f"benchmark plan BrowserGym capture_receipt.{key}",
        )
    production_receipt_policy = (
        browsergym_prompts.PRODUCTION_CAPTURE_RECEIPT_IDENTITY
    )
    if not isinstance(production_receipt_policy, Mapping):
        raise ValueError(
            "production BrowserGym receipt constants must be an object"
        )
    _exact_keys(
        production_receipt_policy,
        _BROWSERGYM_CAPTURE_RECEIPT_POLICY_KEYS,
        label="production BrowserGym receipt constants",
    )
    if (
        production_receipt_policy.get("status")
        != "frozen_controlled_acquisition"
        or any(
            production_receipt_policy.get(key) is None
            for key in _BROWSERGYM_CAPTURE_RECEIPT_KEYS
        )
    ):
        raise ValueError("production BrowserGym receipt constants remain pending")
    if dict(receipt_policy) != dict(production_receipt_policy):
        raise ValueError(
            "benchmark plan BrowserGym capture_receipt disagrees with "
            "production receipt constants"
        )

    receipt = audit.get("capture_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{label}.capture_receipt must be an object")
    _exact_keys(
        receipt,
        _BROWSERGYM_CAPTURE_RECEIPT_KEYS,
        label=f"{label}.capture_receipt",
    )
    if dict(receipt) != expected_receipt:
        raise ValueError(
            f"{label}.capture_receipt disagrees with the frozen BrowserGym plan"
        )
    if receipt.get("file") == capture.get("file"):
        raise ValueError(f"{label} capture and receipt file identities must be distinct")

    source_pins = audit.get("source_pins")
    if not isinstance(source_pins, Mapping):
        raise ValueError(f"{label}.source_pins must be an object")
    _exact_keys(source_pins, _BROWSERGYM_SOURCE_PIN_KEYS, label=f"{label}.source_pins")
    expected_source_pins = {
        "browsergym_revision": suite["revision"],
        "browsergym_version": suite_plan.get("browsergym_version"),
        "miniwob_revision": suite_plan.get("miniwob_revision"),
    }
    if dict(source_pins) != expected_source_pins:
        raise ValueError(f"{label}.source_pins disagree with benchmark plan")

    runtime_pins = audit.get("runtime_pins")
    runtime_policy = suite_plan.get("runtime_pins")
    if not isinstance(runtime_pins, Mapping) or not isinstance(runtime_policy, Mapping):
        raise ValueError(f"{label} BrowserGym runtime pins are missing")
    _exact_keys(runtime_pins, _BROWSERGYM_RUNTIME_PIN_KEYS, label=f"{label}.runtime_pins")
    for key, expected in runtime_policy.items():
        if not _canonical_equal(runtime_pins.get(key), expected):
            raise ValueError(f"{label}.runtime_pins.{key} disagrees with benchmark plan")
    for key in (
        "action_set",
        "architecture",
        "chromium_revision",
        "chromium_version",
        "locale",
        "observation_mode",
        "os",
        "playwright_version",
        "python_version",
        "timezone_id",
    ):
        _nonempty_string(runtime_pins.get(key), label=f"{label}.runtime_pins.{key}")
    if not isinstance(runtime_pins.get("headless"), bool):
        raise ValueError(f"{label}.runtime_pins.headless must be boolean")
    _positive_int(runtime_pins.get("max_steps"), label=f"{label}.runtime_pins.max_steps")
    _positive_finite_number(
        runtime_pins.get("device_scale_factor"),
        label=f"{label}.runtime_pins.device_scale_factor",
    )
    _positive_finite_number(
        runtime_pins.get("playwright_operation_timeout_seconds"),
        label=f"{label}.runtime_pins.playwright_operation_timeout_seconds",
    )
    viewport = runtime_pins.get("viewport")
    if not isinstance(viewport, Mapping):
        raise ValueError(f"{label}.runtime_pins.viewport must be an object")
    _exact_keys(viewport, frozenset({"height", "width"}), label=f"{label}.runtime_pins.viewport")
    _positive_int(viewport.get("height"), label=f"{label}.runtime_pins.viewport.height")
    _positive_int(viewport.get("width"), label=f"{label}.runtime_pins.viewport.width")
    for key in ("browser_executable", "browser_installation"):
        identity = runtime_pins.get(key)
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label}.runtime_pins.{key} must be an object")
        _exact_keys(identity, frozenset({"bytes", "sha256"}), label=f"{label}.{key}")
        _positive_int(identity.get("bytes"), label=f"{label}.{key}.bytes")
        identity_sha256 = identity.get("sha256")
        if (
            not isinstance(identity_sha256, str)
            or _SHA256.fullmatch(identity_sha256) is None
        ):
            raise ValueError(f"{label}.{key}.sha256 is invalid")

    plan = audit.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError(f"{label}.plan must be an object")
    _exact_keys(plan, _BROWSERGYM_PLAN_KEYS, label=f"{label}.plan")
    expected_rows = _positive_int(
        suite_plan.get("expected_prompt_rows"),
        label="benchmark plan BrowserGym expected_prompt_rows",
    )
    expected_variants = _positive_int(
        suite_plan.get("expected_task_variants"),
        label="benchmark plan BrowserGym expected_task_variants",
    )
    expected_groups = _positive_int(
        suite_plan.get("expected_similarity_groups"),
        label="benchmark plan BrowserGym expected_similarity_groups",
    )
    if output_rows != expected_rows or plan.get("episode_rows") != output_rows:
        raise ValueError(f"{label}.plan episode-row accounting mismatch")
    if (
        plan.get("fixed_seeds") != suite_plan.get("fixed_seeds")
        or plan.get("localagent_policy_exclusions")
        != suite_plan.get("localagent_policy_exclusions")
        or plan.get("splits") != [suite["split"]]
        or plan.get("task_variants") != expected_variants
        or plan.get("similarity_group_count") != expected_groups
    ):
        raise ValueError(f"{label}.plan disagrees with BrowserGym suite policy")
    task_groups = plan.get("task_groups")
    if not isinstance(task_groups, Mapping) or len(task_groups) != expected_variants:
        raise ValueError(f"{label}.plan.task_groups is invalid")
    normalized_task_groups = {
        _nonempty_string(task, label=f"{label}.plan.task_groups task"): _nonnegative_int(
            group,
            label=f"{label}.plan.task_groups[{task!r}]",
        )
        for task, group in task_groups.items()
    }
    task_groups_sha256 = _sha256(_canonical_bytes(normalized_task_groups)[:-1])
    if (
        task_groups_sha256 != suite_plan.get("task_groups_sha256")
        or plan.get("grouping_sha256") != task_groups_sha256
        or len(set(normalized_task_groups.values())) != expected_groups
    ):
        raise ValueError(f"{label}.plan task-group fingerprint mismatch")
    expected_similarity_groups: dict[str, list[str]] = {}
    for task, group in sorted(normalized_task_groups.items()):
        expected_similarity_groups.setdefault(str(group), []).append(task)
    if plan.get("similarity_groups") != expected_similarity_groups:
        raise ValueError(f"{label}.plan similarity-group expansion mismatch")


def _validate_weblinx_source_attestation(
    audit: Mapping[str, Any],
    *,
    suite: Mapping[str, Any],
    suite_plan: Mapping[str, Any],
    raw_reference_audit: Mapping[str, Any] | None,
    label: str,
) -> None:
    _, output_rows = _validated_specialized_output(
        audit,
        suite=suite,
        expected_kind=_KNOWN_SUITE_IDENTITIES["weblinx"]["audit_kind"],
        expected_schema_version=1,
        label=label,
        require_self_hash=True,
    )
    if raw_reference_audit is None:
        raise ValueError(
            "WebLINX freeze requires contract-bound raw chat and splits artifacts"
        )
    if audit.get("adapter_version") != suite["adapter"]["version"]:
        raise ValueError(f"{label}.adapter_version disagrees with WebLINX suite")
    label_isolation = audit.get("label_isolation")
    if not isinstance(label_isolation, Mapping):
        raise ValueError(f"{label}.label_isolation must be an object")
    _exact_keys(
        label_isolation,
        _WEBLINX_LABEL_ISOLATION_KEYS,
        label=f"{label}.label_isolation",
    )
    if any(value is not False for value in label_isolation.values()):
        raise ValueError(f"{label}.label_isolation must contain only false flags")

    raw_plan_sources = suite_plan.get("pinned_prompt_sources")
    raw_sources = audit.get("sources")
    if not isinstance(raw_plan_sources, Mapping) or not isinstance(raw_sources, Mapping):
        raise ValueError(f"{label} WebLINX source identities are missing")
    if set(raw_sources) != {"chat", "splits"}:
        raise ValueError(f"{label}.sources must contain exactly chat and splits")
    chat = raw_sources["chat"]
    splits = raw_sources["splits"]
    if not isinstance(chat, Mapping) or not isinstance(splits, Mapping):
        raise ValueError(f"{label}.sources entries must be objects")
    _exact_keys(chat, _WEBLINX_CHAT_SOURCE_KEYS, label=f"{label}.sources.chat")
    _exact_keys(splits, _WEBLINX_SPLITS_SOURCE_KEYS, label=f"{label}.sources.splits")
    expected_chat = raw_plan_sources.get("chat")
    expected_splits = raw_plan_sources.get("splits")
    if not isinstance(expected_chat, Mapping) or not isinstance(expected_splits, Mapping):
        raise ValueError("benchmark plan WebLINX pinned sources are invalid")
    observed_identities = {
        "chat": {
            "bytes": chat.get("bytes"),
            "file": chat.get("name"),
            "sha256": chat.get("sha256"),
        },
        "splits": {
            "bytes": splits.get("bytes"),
            "file": splits.get("name"),
            "sha256": splits.get("sha256"),
        },
    }
    expected_identities = {
        name: {
            "bytes": source.get("bytes"),
            "file": source.get("file"),
            "sha256": source.get("sha256"),
        }
        for name, source in (("chat", expected_chat), ("splits", expected_splits))
    }
    if observed_identities != expected_identities or chat.get("compression") != "gzip":
        raise ValueError(f"{label}.sources disagree with pinned WebLINX identities")

    split_demos = _positive_int(audit.get("split_demos"), label=f"{label}.split_demos")
    source_rows = _positive_int(audit.get("source_rows"), label=f"{label}.source_rows")
    if (
        split_demos != suite_plan.get("expected_demonstrations")
        or source_rows != suite_plan.get("expected_source_rows")
    ):
        raise ValueError(f"{label} WebLINX source counts disagree with benchmark plan")
    privacy = audit.get("privacy")
    receipt = suite_plan.get("privacy_filter_receipt")
    if not isinstance(privacy, Mapping) or not isinstance(receipt, Mapping):
        raise ValueError(f"{label} WebLINX privacy receipt is missing")
    _exact_keys(privacy, _WEBLINX_PRIVACY_KEYS, label=f"{label}.privacy")
    accepted_demos = _nonnegative_int(
        privacy.get("accepted_demos"),
        label=f"{label}.privacy.accepted_demos",
    )
    excluded_demos = _nonnegative_int(
        privacy.get("excluded_demos"),
        label=f"{label}.privacy.excluded_demos",
    )
    excluded_rows = _nonnegative_int(
        privacy.get("excluded_rows"),
        label=f"{label}.privacy.excluded_rows",
    )
    scanned_demos = _positive_int(
        privacy.get("scanned_demos"),
        label=f"{label}.privacy.scanned_demos",
    )
    filter_version = _nonempty_string(
        privacy.get("filter_version"),
        label=f"{label}.privacy.filter_version",
    )
    raw_reason_counts = privacy.get("reason_counts")
    if not isinstance(raw_reason_counts, Mapping):
        raise ValueError(f"{label}.privacy.reason_counts must be an object")
    reason_counts = {
        _nonempty_string(reason, label=f"{label}.privacy reason"): _positive_int(
            count,
            label=f"{label}.privacy.reason_counts[{reason!r}]",
        )
        for reason, count in raw_reason_counts.items()
    }
    expected_receipt_fields = {
        "accepted_demos": accepted_demos,
        "excluded_demos": excluded_demos,
        "excluded_rows": excluded_rows,
        "filter_version": filter_version,
        "reason_counts": dict(sorted(reason_counts.items())),
        "retained_rows": output_rows,
        "scanned_demos": scanned_demos,
    }
    if expected_receipt_fields != dict(receipt):
        raise ValueError(f"{label}.privacy disagrees with benchmark plan receipt")
    if (
        privacy.get("contains_private_heldout_prompts") is not True
        or privacy.get("redistribution_authorized") is not False
        or scanned_demos != split_demos
        or accepted_demos + excluded_demos != split_demos
        or excluded_rows + output_rows != source_rows
    ):
        raise ValueError(f"{label}.privacy accounting or isolation policy mismatch")
    excluded_hashes = privacy.get("excluded_demo_id_sha256")
    if (
        not isinstance(excluded_hashes, list)
        or len(excluded_hashes) != excluded_demos
        or excluded_hashes != sorted(set(excluded_hashes))
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in excluded_hashes
        )
    ):
        raise ValueError(f"{label}.privacy excluded demonstration hashes are invalid")
    excluded_hash_root = privacy.get("excluded_demo_ids_sha256")
    if (
        not isinstance(excluded_hash_root, str)
        or _SHA256.fullmatch(excluded_hash_root) is None
        or _sha256(_canonical_bytes(excluded_hashes)[:-1]) != excluded_hash_root
    ):
        raise ValueError(f"{label}.privacy excluded demonstration hash root mismatch")
    if (
        suite_plan.get("require_whole_demo_sensitive_pattern_exclusion_v1") is not True
        or suite_plan.get("require_private_manual_residual_privacy_review") is not True
        or suite_plan.get("allow_training_use") is not False
    ):
        raise ValueError("benchmark plan WebLINX privacy policy is incompatible")
    if not _canonical_equal(audit, raw_reference_audit):
        raise ValueError(
            f"{label} disagrees with the contract-bound WebLINX raw-source reexport"
        )


def _validate_mind2web_source_attestation(
    audit: Mapping[str, Any],
    *,
    suite: Mapping[str, Any],
    suite_plan: Mapping[str, Any],
    raw_reference_audit: Mapping[str, Any] | None,
    label: str,
) -> None:
    """Require the v2 adapter's protected-archive and ranker chain."""

    if (
        audit.get("kind") != "localagent_mind2web_prompt_adapter_audit"
        or audit.get("schema_version") != 3
        or isinstance(audit.get("schema_version"), bool)
    ):
        raise ValueError(f"{label} must use the Mind2Web v2 archive-bound audit")
    if raw_reference_audit is None:
        raise ValueError(
            "Mind2Web freeze requires a contract-bound archive and ranker config"
        )
    required_audit_identity = {
        "adapter_version": suite["adapter"]["version"],
        "benchmark": suite["benchmark"],
        "mode": "production",
        "revision": suite["revision"],
        "split": suite["split"],
    }
    for key, expected in required_audit_identity.items():
        if audit.get(key) != expected:
            raise ValueError(f"{label}.{key} disagrees with the Mind2Web suite")
    audit_self_sha256 = audit.get("audit_self_sha256")
    if (
        not isinstance(audit_self_sha256, str)
        or _SHA256.fullmatch(audit_self_sha256) is None
    ):
        raise ValueError(f"{label}.audit_self_sha256 is required for Mind2Web v2")

    attestation = audit.get("source_attestation")
    if not isinstance(attestation, Mapping):
        raise ValueError(f"{label}.source_attestation must be an object")
    _exact_keys(
        attestation,
        _MIND2WEB_ATTESTATION_KEYS,
        label=f"{label}.source_attestation",
    )
    if (
        attestation.get("kind")
        != "localagent_mind2web_protected_archive_attestation"
        or attestation.get("schema_version") != 1
        or isinstance(attestation.get("schema_version"), bool)
    ):
        raise ValueError(f"{label}.source_attestation kind/schema is unsupported")

    plan_archive = suite_plan.get("protected_test_archive")
    if not isinstance(plan_archive, Mapping):
        raise ValueError("benchmark plan Mind2Web protected_test_archive is missing")
    if (
        plan_archive.get("require_exact_member_set") is not True
        or plan_archive.get("require_plaintext_member_hash_binding") is not True
    ):
        raise ValueError("benchmark plan Mind2Web archive-binding policy is incompatible")
    raw_plan_member_splits = plan_archive.get("member_splits")
    if not isinstance(raw_plan_member_splits, Mapping):
        raise ValueError("benchmark plan Mind2Web member_splits is missing")
    plan_member_splits = {
        _safe_posix_member_name(
            member,
            label="benchmark plan Mind2Web member_splits key",
        ): _nonempty_string(
            split,
            label=f"benchmark plan Mind2Web member_splits[{member!r}]",
        )
        for member, split in raw_plan_member_splits.items()
    }

    archive = attestation.get("archive")
    if not isinstance(archive, Mapping):
        raise ValueError(f"{label}.source_attestation.archive must be an object")
    _exact_keys(
        archive,
        _MIND2WEB_ARCHIVE_KEYS,
        label=f"{label}.source_attestation.archive",
    )
    archive_bytes = _positive_int(
        archive.get("bytes"),
        label=f"{label}.source_attestation.archive.bytes",
    )
    archive_sha256 = archive.get("sha256")
    if not isinstance(archive_sha256, str) or _SHA256.fullmatch(archive_sha256) is None:
        raise ValueError(f"{label}.source_attestation.archive.sha256 is invalid")
    _nonempty_string(
        archive.get("name"),
        label=f"{label}.source_attestation.archive.name",
    )
    if (
        archive_bytes != plan_archive.get("bytes")
        or archive_sha256 != plan_archive.get("sha256")
    ):
        raise ValueError(f"{label}.source_attestation archive disagrees with benchmark plan")

    archive_format = attestation.get("archive_format")
    if not isinstance(archive_format, Mapping):
        raise ValueError(f"{label}.source_attestation.archive_format must be an object")
    _exact_keys(
        archive_format,
        _MIND2WEB_ARCHIVE_FORMAT_KEYS,
        label=f"{label}.source_attestation.archive_format",
    )
    member_count = _positive_int(
        archive_format.get("members"),
        label=f"{label}.source_attestation.archive_format.members",
    )
    if (
        archive_format.get("compression") != plan_archive.get("compression")
        or archive_format.get("encryption") != plan_archive.get("encryption")
        or member_count != plan_archive.get("members")
        or member_count != len(plan_member_splits)
    ):
        raise ValueError(
            f"{label}.source_attestation archive format disagrees with benchmark plan"
        )

    raw_members = attestation.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != member_count:
        raise ValueError(f"{label}.source_attestation.members count mismatch")
    member_names: list[str] = []
    member_bindings: dict[str, dict[str, Any]] = {}
    tasks_by_split: Counter[str] = Counter()
    total_member_bytes = 0
    total_member_rows = 0
    for index, member in enumerate(raw_members):
        member_label = f"{label}.source_attestation.members[{index}]"
        if not isinstance(member, Mapping):
            raise ValueError(f"{member_label} must be an object")
        _exact_keys(member, _MIND2WEB_MEMBER_KEYS, label=member_label)
        member_name = _safe_posix_member_name(
            member.get("member"),
            label=f"{member_label}.member",
        )
        split = _nonempty_string(member.get("split"), label=f"{member_label}.split")
        member_names.append(member_name)
        member_bytes = _positive_int(
            member.get("bytes"),
            label=f"{member_label}.bytes",
        )
        _positive_int(
            member.get("compressed_bytes"),
            label=f"{member_label}.compressed_bytes",
        )
        tasks = _positive_int(member.get("tasks"), label=f"{member_label}.tasks")
        rows = _positive_int(member.get("rows"), label=f"{member_label}.rows")
        if rows < tasks:
            raise ValueError(f"{member_label}.rows must be at least its task count")
        member_sha256 = member.get("sha256")
        if (
            not isinstance(member_sha256, str)
            or _SHA256.fullmatch(member_sha256) is None
        ):
            raise ValueError(f"{member_label}.sha256 is invalid")
        crc32 = member.get("crc32")
        if not isinstance(crc32, str) or _CRC32.fullmatch(crc32) is None:
            raise ValueError(f"{member_label}.crc32 is invalid")
        tasks_by_split[split] += tasks
        total_member_bytes += member_bytes
        total_member_rows += rows
        member_bindings[member_name] = {
            "archive_member": member_name,
            "bytes": member_bytes,
            "rows": rows,
            "sha256": member_sha256,
            "split": split,
            "tasks": tasks,
        }
    if member_names != sorted(member_names) or len(member_names) != len(set(member_names)):
        raise ValueError(f"{label}.source_attestation.members must be uniquely sorted")
    observed_member_splits = {
        member: str(binding["split"])
        for member, binding in member_bindings.items()
    }
    if observed_member_splits != plan_member_splits:
        raise ValueError(
            f"{label}.source_attestation member layout disagrees with benchmark plan"
        )

    members_sha256 = attestation.get("members_sha256")
    if (
        not isinstance(members_sha256, str)
        or _SHA256.fullmatch(members_sha256) is None
        or _sha256(_canonical_bytes(raw_members)[:-1]) != members_sha256
    ):
        raise ValueError(f"{label}.source_attestation.members_sha256 mismatch")
    raw_tasks_by_split = attestation.get("tasks_by_split")
    expected_tasks = suite_plan.get("heldout_splits")
    if not isinstance(raw_tasks_by_split, Mapping) or not isinstance(
        expected_tasks,
        Mapping,
    ):
        raise ValueError(f"{label}.source_attestation tasks_by_split is invalid")
    normalized_tasks = {
        _nonempty_string(key, label=f"{label}.source_attestation split"): _positive_int(
            value,
            label=f"{label}.source_attestation.tasks_by_split[{key!r}]",
        )
        for key, value in raw_tasks_by_split.items()
    }
    if normalized_tasks != dict(expected_tasks) or dict(tasks_by_split) != normalized_tasks:
        raise ValueError(
            f"{label}.source_attestation task counts disagree with benchmark plan"
        )
    total_tasks = _positive_int(
        attestation.get("total_tasks"),
        label=f"{label}.source_attestation.total_tasks",
    )
    audit_tasks = _positive_int(audit.get("tasks"), label=f"{label}.tasks")
    if total_tasks != sum(normalized_tasks.values()) or total_tasks != audit_tasks:
        raise ValueError(f"{label}.source_attestation.total_tasks mismatch")
    total_uncompressed_bytes = _positive_int(
        attestation.get("total_uncompressed_bytes"),
        label=f"{label}.source_attestation.total_uncompressed_bytes",
    )
    if (
        total_uncompressed_bytes != total_member_bytes
        or total_uncompressed_bytes != plan_archive.get("uncompressed_bytes")
    ):
        raise ValueError(
            f"{label}.source_attestation uncompressed bytes disagree with benchmark plan"
        )

    raw_sources = audit.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != member_count:
        raise ValueError(f"{label}.sources must bind every protected archive member")
    source_names: list[str] = []
    source_bindings: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(raw_sources):
        source_label = f"{label}.sources[{index}]"
        if not isinstance(source, Mapping):
            raise ValueError(f"{source_label} must be an object")
        _exact_keys(source, _MIND2WEB_SOURCE_KEYS, label=source_label)
        archive_member = _safe_posix_member_name(
            source.get("archive_member"),
            label=f"{source_label}.archive_member",
        )
        source_names.append(archive_member)
        source_sha256 = source.get("sha256")
        if (
            not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
        ):
            raise ValueError(f"{source_label}.sha256 is invalid")
        source_bindings[archive_member] = {
            "archive_member": archive_member,
            "bytes": _positive_int(source.get("bytes"), label=f"{source_label}.bytes"),
            "rows": _positive_int(source.get("rows"), label=f"{source_label}.rows"),
            "sha256": source_sha256,
            "split": _nonempty_string(
                source.get("split"),
                label=f"{source_label}.split",
            ),
            "tasks": _positive_int(
                source.get("tasks"),
                label=f"{source_label}.tasks",
            ),
        }
        _nonempty_string(source.get("name"), label=f"{source_label}.name")
    if source_names != member_names or source_bindings != member_bindings:
        raise ValueError(
            f"{label}.sources disagree with protected archive member attestation"
        )

    output = audit.get("output")
    freeze_binding = audit.get("freeze_binding")
    if not isinstance(output, Mapping) or not isinstance(freeze_binding, Mapping):
        raise ValueError(f"{label} output/freeze binding is invalid")
    binding_output = freeze_binding.get("output")
    if not isinstance(binding_output, Mapping):
        raise ValueError(f"{label}.freeze_binding.output is invalid")
    output_rows = _positive_int(output.get("rows"), label=f"{label}.output.rows")
    if (
        output_rows != total_member_rows
        or output_rows != binding_output.get("records")
        or output.get("bytes") != binding_output.get("bytes")
        or output.get("sha256") != binding_output.get("sha256")
    ):
        raise ValueError(f"{label} member rows and bound prompt output disagree")

    ranker_policy = suite_plan.get("prompt_ranker")
    if not isinstance(ranker_policy, Mapping):
        raise ValueError("benchmark plan Mind2Web prompt_ranker is missing")
    _exact_keys(
        ranker_policy,
        _MIND2WEB_RANKER_POLICY_KEYS,
        label="benchmark plan Mind2Web prompt_ranker",
    )
    if ranker_policy.get("adapter_version") != suite["adapter"]["version"]:
        raise ValueError(
            "benchmark plan Mind2Web ranker adapter version disagrees with suite"
        )
    config_policy = ranker_policy.get("config")
    implementation_policy = ranker_policy.get("implementation")
    adapter_implementation_policy = ranker_policy.get("adapter_implementation")
    runtime_policy = ranker_policy.get("runtime")
    budget_policy = ranker_policy.get("budget")
    projection_policy = ranker_policy.get("input_projection")
    policy_shapes = (
        (
            config_policy,
            _MIND2WEB_RANKER_CONFIG_POLICY_KEYS,
            "benchmark plan Mind2Web prompt_ranker.config",
        ),
        (
            implementation_policy,
            _MIND2WEB_IMPLEMENTATION_POLICY_KEYS,
            "benchmark plan Mind2Web prompt_ranker.implementation",
        ),
        (
            adapter_implementation_policy,
            _MIND2WEB_IMPLEMENTATION_POLICY_KEYS,
            "benchmark plan Mind2Web prompt_ranker.adapter_implementation",
        ),
        (
            runtime_policy,
            _MIND2WEB_RUNTIME_POLICY_KEYS,
            "benchmark plan Mind2Web prompt_ranker.runtime",
        ),
        (
            budget_policy,
            _MIND2WEB_RANKER_BUDGET_POLICY_KEYS,
            "benchmark plan Mind2Web prompt_ranker.budget",
        ),
        (
            projection_policy,
            _MIND2WEB_INPUT_PROJECTION_KEYS,
            "benchmark plan Mind2Web prompt_ranker.input_projection",
        ),
    )
    for value, keys, policy_label in policy_shapes:
        if not isinstance(value, Mapping):
            raise ValueError(f"{policy_label} must be an object")
        _exact_keys(value, keys, label=policy_label)

    config_sha256 = config_policy.get("sha256")
    config_self_sha256 = config_policy.get("config_self_sha256")
    for value, field in (
        (config_sha256, "sha256"),
        (config_self_sha256, "config_self_sha256"),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(
                f"benchmark plan Mind2Web prompt_ranker.config.{field} is invalid"
            )
    expected_ranker_identity = {
        "artifact": {
            "bytes": _positive_int(
                config_policy.get("bytes"),
                label="benchmark plan Mind2Web prompt_ranker.config.bytes",
            ),
            "config_self_sha256": config_self_sha256,
            "name": _nonempty_string(
                config_policy.get("file"),
                label="benchmark plan Mind2Web prompt_ranker.config.file",
            ),
            "sha256": config_sha256,
        },
        "implementation": dict(implementation_policy),
        "input_projection": dict(projection_policy),
        "ranker_version": _nonempty_string(
            ranker_policy.get("ranker_version"),
            label="benchmark plan Mind2Web prompt_ranker.ranker_version",
        ),
        "runtime": dict(runtime_policy),
    }
    ranking = audit.get("ranking")
    if not isinstance(ranking, Mapping):
        raise ValueError(f"{label}.ranking must be an object")
    if not _canonical_equal(ranking.get("ranker"), expected_ranker_identity):
        raise ValueError(f"{label}.ranking.ranker disagrees with benchmark plan")
    if not _canonical_equal(
        ranking.get("adapter_implementation"),
        adapter_implementation_policy,
    ):
        raise ValueError(
            f"{label}.ranking.adapter_implementation disagrees with benchmark plan"
        )
    if not _canonical_equal(ranking.get("budget"), budget_policy):
        raise ValueError(f"{label}.ranking.budget disagrees with benchmark plan")
    if not _canonical_equal(ranking.get("input_projection"), projection_policy):
        raise ValueError(
            f"{label}.ranking.input_projection disagrees with benchmark plan"
        )
    dependencies = ranking.get("dependencies")
    if (
        not isinstance(dependencies, Mapping)
        or not dependencies
        or any(value is not False for value in dependencies.values())
    ):
        raise ValueError(f"{label}.ranking dependencies must all be false")
    label_isolation = audit.get("label_isolation")
    if (
        not isinstance(label_isolation, Mapping)
        or not label_isolation
        or any(value is not False for value in label_isolation.values())
    ):
        raise ValueError(f"{label}.label_isolation must contain only false flags")
    if (
        not _canonical_equal(freeze_binding.get("ranker"), expected_ranker_identity)
        or not _canonical_equal(
            freeze_binding.get("adapter_implementation"),
            adapter_implementation_policy,
        )
    ):
        raise ValueError(f"{label}.freeze_binding ranker/code identity mismatch")

    # Perform this comparison last so malformed semantic receipts report their precise cause.
    if not _canonical_equal(audit, raw_reference_audit):
        raise ValueError(
            f"{label} disagrees with the contract-bound Mind2Web raw-source reexport"
        )


def _verified_adapter_provenance(
    artifact: _Artifact,
    *,
    max_bytes: int,
    suite: Mapping[str, Any],
    suite_plan: Mapping[str, Any],
    source_identities: Sequence[Mapping[str, Any]],
    bfcl_raw_reference_audit: Mapping[str, Any] | None,
    browsergym_raw_reference_audit: Mapping[str, Any] | None,
    mind2web_raw_reference_audit: Mapping[str, Any] | None,
    weblinx_raw_reference_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = _verified_artifact(
        artifact,
        max_bytes=max_bytes,
        artifact_kind="adapter provenance",
    )
    payload = _read_bounded_regular_file(
        artifact.path,
        max_bytes=max_bytes,
        label=f"adapter provenance {artifact.name!r}",
    )
    if len(payload) != identity["bytes"] or _sha256(payload) != identity["sha256"]:
        raise ValueError(
            f"adapter provenance {artifact.name!r} changed while it was being verified"
        )
    audit = _strict_json_loads(payload, label=f"adapter provenance {artifact.name!r}")
    if not isinstance(audit, Mapping):
        raise ValueError(f"adapter provenance {artifact.name!r} must be a JSON object")
    kind = _nonempty_string(
        audit.get("kind"),
        label=f"adapter provenance {artifact.name!r}.kind",
    )
    schema_version = _positive_int(
        audit.get("schema_version"),
        label=f"adapter provenance {artifact.name!r}.schema_version",
    )
    adapter = _nonempty_string(
        audit.get("adapter"),
        label=f"adapter provenance {artifact.name!r}.adapter",
    )
    declared_adapter = suite["adapter"]
    if adapter != declared_adapter["version"]:
        raise ValueError(
            f"adapter provenance {artifact.name!r} adapter disagrees with suite"
        )
    purpose = _nonempty_string(
        audit.get("purpose"),
        label=f"adapter provenance {artifact.name!r}.purpose",
    )
    if "prompt" not in purpose.casefold() or "decontamination" not in purpose.casefold():
        raise ValueError(
            f"adapter provenance {artifact.name!r} purpose is not prompt decontamination"
        )
    revision = audit.get("revision")
    if revision is not None and revision != suite["revision"]:
        raise ValueError(
            f"adapter provenance {artifact.name!r} revision disagrees with suite"
        )
    output = audit.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"adapter provenance {artifact.name!r}.output must be an object")
    output_bytes = _positive_int(
        output.get("bytes"),
        label=f"adapter provenance {artifact.name!r}.output.bytes",
    )
    output_sha256 = output.get("sha256")
    if not isinstance(output_sha256, str) or _SHA256.fullmatch(output_sha256) is None:
        raise ValueError(
            f"adapter provenance {artifact.name!r}.output.sha256 must be lowercase SHA-256"
        )
    if not any(
        source.get("bytes") == output_bytes
        and source.get("sha256") == output_sha256
        for source in source_identities
    ):
        raise ValueError(
            f"adapter provenance {artifact.name!r} output is not a declared prompt source"
        )

    freeze_binding = audit.get("freeze_binding")
    if not isinstance(freeze_binding, Mapping):
        raise ValueError(
            f"adapter provenance {artifact.name!r}.freeze_binding must be an object"
        )
    _exact_keys(
        freeze_binding,
        (
            _MIND2WEB_FREEZE_BINDING_KEYS
            if _is_mind2web_suite(suite)
            else _FREEZE_BINDING_KEYS
        ),
        label=f"adapter provenance {artifact.name!r}.freeze_binding",
    )
    binding_output = freeze_binding.get("output")
    if not isinstance(binding_output, Mapping):
        raise ValueError(
            f"adapter provenance {artifact.name!r}.freeze_binding.output must be an object"
        )
    _exact_keys(
        binding_output,
        _FREEZE_BINDING_OUTPUT_KEYS,
        label=f"adapter provenance {artifact.name!r}.freeze_binding.output",
    )
    binding_records = _positive_int(
        binding_output.get("records"),
        label=f"adapter provenance {artifact.name!r}.freeze_binding.output.records",
    )
    expected_binding = {
        "adapter": declared_adapter["version"],
        "benchmark": suite["benchmark"],
        "mode": "production",
        "revision": suite["revision"],
        "split": suite["split"],
        "prompt_only": True,
        "contains_current_step_labels": False,
        "output": {
            "bytes": output_bytes,
            "sha256": output_sha256,
            "records": binding_records,
        },
    }
    if _is_mind2web_suite(suite):
        ranking = audit.get("ranking")
        if not isinstance(ranking, Mapping):
            raise ValueError(
                f"adapter provenance {artifact.name!r}.ranking must be an object"
            )
        expected_binding["ranker"] = ranking.get("ranker")
        expected_binding["adapter_implementation"] = ranking.get(
            "adapter_implementation"
        )
    if dict(freeze_binding) != expected_binding:
        raise ValueError(
            f"adapter provenance {artifact.name!r} freeze_binding disagrees with suite"
        )
    if not any(
        source.get("bytes") == output_bytes
        and source.get("sha256") == output_sha256
        and source.get("records") == binding_records
        for source in source_identities
    ):
        raise ValueError(
            f"adapter provenance {artifact.name!r} freeze_binding record count "
            "is not a declared prompt source"
        )

    observed_suite_identity = {
        "adapter": declared_adapter["version"],
        "benchmark": suite["benchmark"],
        "name": suite["name"],
    }
    specialized_suite: str | None = None
    for suite_name, expected_identity in _KNOWN_SUITE_IDENTITIES.items():
        matched_fields = sum(
            observed_suite_identity[key] == expected_identity[key]
            for key in ("adapter", "benchmark", "name")
        )
        marked = kind == expected_identity["audit_kind"] or matched_fields > 0
        if not marked:
            continue
        if matched_fields != 3:
            raise ValueError(
                f"adapter provenance {artifact.name!r} uses a partial "
                f"{suite_name} production identity"
            )
        if specialized_suite is not None and specialized_suite != suite_name:
            raise ValueError(
                f"adapter provenance {artifact.name!r} mixes known suite identities"
            )
        specialized_suite = suite_name

    if specialized_suite == "bfcl":
        _validate_bfcl_source_attestation(
            audit,
            suite=suite,
            suite_plan=suite_plan,
            raw_reference_audit=bfcl_raw_reference_audit,
            label=f"adapter provenance {artifact.name!r}",
        )
    elif specialized_suite == "browsergym":
        _validate_browsergym_source_attestation(
            audit,
            suite=suite,
            suite_plan=suite_plan,
            raw_reference_audit=browsergym_raw_reference_audit,
            label=f"adapter provenance {artifact.name!r}",
        )
    elif specialized_suite == "mind2web":
        _validate_mind2web_source_attestation(
            audit,
            suite=suite,
            suite_plan=suite_plan,
            raw_reference_audit=mind2web_raw_reference_audit,
            label=f"adapter provenance {artifact.name!r}",
        )
    elif specialized_suite == "weblinx":
        _validate_weblinx_source_attestation(
            audit,
            suite=suite,
            suite_plan=suite_plan,
            raw_reference_audit=weblinx_raw_reference_audit,
            label=f"adapter provenance {artifact.name!r}",
        )

    audit_self_sha256 = audit.get("audit_self_sha256")
    if audit_self_sha256 is not None:
        if (
            not isinstance(audit_self_sha256, str)
            or _SHA256.fullmatch(audit_self_sha256) is None
        ):
            raise ValueError(
                f"adapter provenance {artifact.name!r}.audit_self_sha256 is invalid"
            )
        without_hash = dict(audit)
        without_hash.pop("audit_self_sha256")
        canonical_without_newline = _canonical_bytes(without_hash)[:-1]
        if _sha256(canonical_without_newline) != audit_self_sha256:
            raise ValueError(
                f"adapter provenance {artifact.name!r} self-hash mismatch"
            )
    return {
        **identity,
        "adapter": adapter,
        "audit_kind": kind,
        "audit_schema_version": schema_version,
        "bound_prompt_source": {
            "bytes": output_bytes,
            "records": binding_records,
            "sha256": output_sha256,
        },
    }


def _source_rows(
    artifact: _Artifact,
    *,
    suite: Mapping[str, Any],
    limits: Mapping[str, int],
    observed_source_ids: set[str],
    remaining_rows: int,
) -> tuple[list[_PromptRow], dict[str, Any]]:
    if artifact.path.stat().st_size != artifact.bytes:
        raise ValueError(f"source {artifact.name!r} byte identity disagrees with contract")
    rows: list[_PromptRow] = []
    observed_bytes = 0
    digest = hashlib.sha256()
    line_number = 0
    with artifact.path.open("rb") as handle:
        while True:
            raw = handle.readline(limits["max_record_bytes"] + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > limits["max_record_bytes"]:
                raise ValueError(
                    f"source {artifact.name!r}:{line_number} exceeds max_record_bytes"
                )
            observed_bytes += len(raw)
            if observed_bytes > limits["max_source_bytes"]:
                raise ValueError(f"source {artifact.name!r} exceeds max_source_bytes")
            digest.update(raw)
            if not raw.strip():
                raise ValueError(
                    f"source {artifact.name!r}:{line_number} must not be blank"
                )
            if len(rows) >= remaining_rows:
                raise ValueError("source rows exceed limits.max_rows")
            label = f"source {artifact.name!r}:{line_number}"
            record = _strict_json_loads(raw, label=label)
            if not isinstance(record, Mapping):
                raise ValueError(f"{label} must be a JSON object")
            _reject_label_fields(record, label=label)
            _exact_keys(record, _ROW_KEYS, label=label)
            source_case_id = _nonempty_string(
                record.get("source_case_id"), label=f"{label}.source_case_id"
            )
            if len(source_case_id.encode("utf-8")) > _MAX_SOURCE_CASE_ID_BYTES:
                raise ValueError(
                    f"{label}.source_case_id exceeds {_MAX_SOURCE_CASE_ID_BYTES} bytes"
                )
            prompt = _nonempty_string(record.get("prompt"), label=f"{label}.prompt")
            prompt_bytes = len(prompt.encode("utf-8"))
            if prompt_bytes > min(_MAX_PROMPT_BYTES, limits["max_record_bytes"]):
                raise ValueError(f"{label}.prompt exceeds the bounded prompt size")
            normalized = _normalized_prompt(prompt)
            if not normalized:
                raise ValueError(f"{label}.prompt is empty after normalization")
            source_id_hash = _source_case_id_sha256(source_case_id, suite=suite)
            if source_id_hash in observed_source_ids:
                raise ValueError(f"duplicate source_case_id at {label}")
            observed_source_ids.add(source_id_hash)
            rows.append(
                _PromptRow(
                    source_case_id_sha256=source_id_hash,
                    prompt=prompt,
                    normalized_prompt=normalized,
                )
            )

    observed_sha256 = digest.hexdigest()
    if observed_bytes != artifact.bytes or observed_sha256 != artifact.sha256:
        raise ValueError(f"source {artifact.name!r} byte identity disagrees with contract")
    if len(rows) != artifact.records:
        raise ValueError(
            f"source {artifact.name!r} record-count mismatch: "
            f"expected {artifact.records}, got {len(rows)}"
        )
    return rows, {
        "name": artifact.name,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "records": len(rows),
    }


def _deduplicated_output(
    rows: Sequence[_PromptRow],
) -> tuple[bytes, dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.normalized_prompt,
            row.prompt,
            row.source_case_id_sha256,
        ),
    )
    selected: list[_PromptRow] = []
    seen_prompts: set[str] = set()
    for row in ordered:
        if row.normalized_prompt in seen_prompts:
            continue
        seen_prompts.add(row.normalized_prompt)
        selected.append(row)

    payload = b"".join(
        _canonical_bytes(
            {
                "prompt": row.prompt,
                "source_case_id_sha256": row.source_case_id_sha256,
            }
        )
        for row in selected
    )
    if len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError(f"canonical denylist output exceeds {_MAX_OUTPUT_BYTES} bytes")
    normalized_prompts = [row.normalized_prompt for row in selected]
    source_hashes = sorted(row.source_case_id_sha256 for row in rows)
    return payload, {
        "method": "unicode_nfkc_casefold_token_normalization_v1",
        "input_rows": len(rows),
        "unique_normalized_prompts": len(selected),
        "normalized_prompt_duplicates_removed": len(rows) - len(selected),
        "normalized_prompt_set_sha256": _joined_fingerprint(normalized_prompts),
        "input_source_case_id_hashes_sha256": _joined_fingerprint(source_hashes),
        "representative": (
            "lexicographically smallest exact prompt then hashed source ID "
            "within each normalized-prompt group"
        ),
    }


def _assert_existing_or_absent(path: Path, payload: bytes) -> None:
    if path.exists():
        if not _matches_payload(path, payload):
            raise RuntimeError(f"refusing to overwrite drifted frozen artifact: {path}")


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        if not _matches_payload(path, payload):
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
            os.link(temporary, path)
        except FileExistsError:
            if not _matches_payload(path, payload):
                raise RuntimeError(
                    f"refusing to overwrite concurrently created artifact: {path}"
                )
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    if not _matches_payload(path, payload):
        raise RuntimeError(f"published artifact failed byte verification: {path}")


def freeze_evaluation_denylist_suite(
    contract_path: str | Path,
    *,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify and freeze one prompt-only evaluation denylist suite.

    The contract and adapter schema are intentionally strict.  Adapter rows must contain exactly
    ``source_case_id`` and ``prompt``.  Existing outputs are accepted only when byte-identical to
    the newly derived canonical artifacts.
    """

    contract_file = Path(contract_path)
    contract_payload = _read_bounded_regular_file(
        contract_file,
        max_bytes=_MAX_CONTRACT_BYTES,
        label="contract",
    )
    contract = _strict_json_loads(contract_payload, label=str(contract_file))
    if not isinstance(contract, Mapping):
        raise ValueError("contract must be a JSON object")
    contract_keys = set(contract)
    missing_contract_keys = sorted(_CONTRACT_REQUIRED_KEYS - contract_keys)
    extra_contract_keys = sorted(
        contract_keys - _CONTRACT_REQUIRED_KEYS - _CONTRACT_OPTIONAL_KEYS
    )
    if missing_contract_keys or extra_contract_keys:
        raise ValueError(
            "contract keys mismatch: "
            f"missing={missing_contract_keys}, extra={extra_contract_keys}"
        )
    if contract.get("kind") != CONTRACT_KIND or contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"contract must be {CONTRACT_KIND!r} schema_version {SCHEMA_VERSION}"
        )
    suite = _validated_suite(contract.get("suite"))
    limits = _validated_limits(contract.get("limits"))

    raw_benchmark_plan = contract.get("benchmark_plan")
    benchmark_plan = _artifact_from_record(
        raw_benchmark_plan,
        base=contract_file.parent,
        label="benchmark_plan",
        source=False,
    )
    if benchmark_plan.bytes > limits["max_benchmark_plan_bytes"]:
        raise ValueError(
            "declared benchmark plan exceeds limits.max_benchmark_plan_bytes"
        )

    raw_sources = contract.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("sources must be a non-empty array")
    if len(raw_sources) > _MAX_SOURCE_ARTIFACTS:
        raise ValueError(f"sources exceeds hard artifact cap {_MAX_SOURCE_ARTIFACTS}")
    sources = [
        _artifact_from_record(
            raw,
            base=contract_file.parent,
            label=f"sources[{index}]",
            source=True,
        )
        for index, raw in enumerate(raw_sources)
    ]

    raw_adapter_provenance = contract.get("adapter_provenance")
    if not isinstance(raw_adapter_provenance, list) or not raw_adapter_provenance:
        raise ValueError("adapter_provenance must be a non-empty array")
    if len(raw_adapter_provenance) > _MAX_ADAPTER_PROVENANCE_ARTIFACTS:
        raise ValueError(
            "adapter_provenance exceeds hard artifact cap "
            f"{_MAX_ADAPTER_PROVENANCE_ARTIFACTS}"
        )
    adapter_provenance = [
        _artifact_from_record(
            raw,
            base=contract_file.parent,
            label=f"adapter_provenance[{index}]",
            source=False,
        )
        for index, raw in enumerate(raw_adapter_provenance)
    ]

    raw_evidence = contract.get("license_evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("license_evidence must be a non-empty array")
    if len(raw_evidence) > _MAX_LICENSE_EVIDENCE_ARTIFACTS:
        raise ValueError(
            "license_evidence exceeds hard artifact cap "
            f"{_MAX_LICENSE_EVIDENCE_ARTIFACTS}"
        )
    evidence = [
        _artifact_from_record(
            raw,
            base=contract_file.parent,
            label=f"license_evidence[{index}]",
            source=False,
        )
        for index, raw in enumerate(raw_evidence)
    ]

    raw_raw_artifacts = contract.get("raw_artifacts", [])
    if not isinstance(raw_raw_artifacts, list):
        raise ValueError("raw_artifacts must be an array")
    if len(raw_raw_artifacts) > _RAW_ARTIFACT_HARD_CAP:
        raise ValueError("raw_artifacts exceeds the hard artifact cap")
    raw_artifacts = [
        _raw_artifact_from_record(
            raw,
            base=contract_file.parent,
            label=f"raw_artifacts[{index}]",
        )
        for index, raw in enumerate(raw_raw_artifacts)
    ]
    if raw_artifacts and not (
        _is_bfcl_suite(suite)
        or _is_browsergym_suite(suite)
        or _is_mind2web_suite(suite)
        or _is_weblinx_suite(suite)
    ):
        raise ValueError(
            "raw_artifacts are supported only for the BFCL, BrowserGym, "
            "Mind2Web, and WebLINX suites"
        )

    all_artifacts = [
        benchmark_plan,
        *sources,
        *adapter_provenance,
        *evidence,
        *(raw.artifact for raw in raw_artifacts),
    ]
    names = [artifact.name for artifact in all_artifacts]
    if len(names) != len(set(names)):
        raise ValueError(
            "all declared input artifact names must be unique"
        )
    resolved_inputs = [artifact.path.resolve() for artifact in all_artifacts]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError(
            "all declared input artifact paths must be distinct"
        )
    if contract_file.resolve() in set(resolved_inputs):
        raise ValueError("contract must be distinct from every declared input artifact")

    if any(artifact.bytes > limits["max_source_bytes"] for artifact in sources):
        raise ValueError("declared source exceeds limits.max_source_bytes")
    total_source_bytes = sum(artifact.bytes for artifact in sources)
    if total_source_bytes > _MAX_TOTAL_SOURCE_BYTES:
        raise ValueError(
            f"declared sources exceed hard total byte cap {_MAX_TOTAL_SOURCE_BYTES}"
        )
    if any(
        artifact.bytes > limits["max_adapter_provenance_bytes"]
        for artifact in adapter_provenance
    ):
        raise ValueError(
            "declared adapter provenance exceeds limits.max_adapter_provenance_bytes"
        )
    total_adapter_provenance_bytes = sum(
        artifact.bytes for artifact in adapter_provenance
    )
    if total_adapter_provenance_bytes > _MAX_TOTAL_ADAPTER_PROVENANCE_BYTES:
        raise ValueError(
            "declared adapter provenance exceeds hard total byte cap "
            f"{_MAX_TOTAL_ADAPTER_PROVENANCE_BYTES}"
        )
    if any(
        artifact.bytes > limits["max_license_evidence_bytes"]
        for artifact in evidence
    ):
        raise ValueError(
            "declared license evidence exceeds limits.max_license_evidence_bytes"
        )
    total_evidence_bytes = sum(artifact.bytes for artifact in evidence)
    if total_evidence_bytes > _MAX_TOTAL_LICENSE_EVIDENCE_BYTES:
        raise ValueError(
            "declared license evidence exceeds hard total byte cap "
            f"{_MAX_TOTAL_LICENSE_EVIDENCE_BYTES}"
        )
    expected_rows = sum(int(artifact.records or 0) for artifact in sources)
    if expected_rows > limits["max_rows"]:
        raise ValueError("declared source records exceed limits.max_rows")

    output_file = Path(output_path)
    manifest_file = Path(manifest_path)
    if output_file.suffix.casefold() not in {".jsonl", ".ndjson"}:
        raise ValueError("output_path must use a .jsonl or .ndjson suffix")
    if output_file.resolve() == manifest_file.resolve():
        raise ValueError("output_path and manifest_path must be distinct")
    protected = {contract_file.resolve(), *resolved_inputs}
    if {output_file.resolve(), manifest_file.resolve()} & protected:
        raise ValueError("frozen outputs must not overwrite contract or input artifacts")

    observed_source_ids: set[str] = set()
    rows: list[_PromptRow] = []
    source_identities: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda artifact: artifact.name):
        source_rows, identity = _source_rows(
            source,
            suite=suite,
            limits=limits,
            observed_source_ids=observed_source_ids,
            remaining_rows=limits["max_rows"] - len(rows),
        )
        rows.extend(source_rows)
        source_identities.append(identity)
    if len(rows) != expected_rows:
        raise ValueError(
            f"total source record-count mismatch: expected {expected_rows}, got {len(rows)}"
        )

    benchmark_plan_identity, benchmark_suite_plan = _verified_benchmark_plan(
        benchmark_plan,
        max_bytes=limits["max_benchmark_plan_bytes"],
        suite=suite,
    )
    raw_artifact_identities: list[dict[str, Any]] = []
    bfcl_raw_reference_audit: Mapping[str, Any] | None = None
    browsergym_raw_reference_audit: Mapping[str, Any] | None = None
    mind2web_raw_reference_audit: Mapping[str, Any] | None = None
    weblinx_raw_reference_audit: Mapping[str, Any] | None = None
    if _is_bfcl_suite(suite):
        (
            raw_artifact_identities,
            bfcl_raw_reference_audit,
        ) = _verified_bfcl_raw_chain(
            raw_artifacts,
            sources=sources,
        )
    elif _is_browsergym_suite(suite):
        (
            raw_artifact_identities,
            browsergym_raw_reference_audit,
        ) = _verified_browsergym_raw_chain(
            raw_artifacts,
            sources=sources,
        )
    elif _is_mind2web_suite(suite):
        (
            raw_artifact_identities,
            mind2web_raw_reference_audit,
        ) = _verified_mind2web_raw_chain(
            raw_artifacts,
            sources=sources,
            suite=suite,
        )
    elif _is_weblinx_suite(suite):
        (
            raw_artifact_identities,
            weblinx_raw_reference_audit,
        ) = _verified_weblinx_raw_chain(
            raw_artifacts,
            sources=sources,
            suite=suite,
        )
    adapter_provenance_identities = [
        _verified_adapter_provenance(
            artifact,
            max_bytes=limits["max_adapter_provenance_bytes"],
            suite=suite,
            suite_plan=benchmark_suite_plan,
            source_identities=source_identities,
            bfcl_raw_reference_audit=bfcl_raw_reference_audit,
            browsergym_raw_reference_audit=browsergym_raw_reference_audit,
            mind2web_raw_reference_audit=mind2web_raw_reference_audit,
            weblinx_raw_reference_audit=weblinx_raw_reference_audit,
        )
        for artifact in sorted(
            adapter_provenance,
            key=lambda artifact: artifact.name,
        )
    ]
    declared_source_bindings = Counter(
        (
            int(source["bytes"]),
            str(source["sha256"]),
            int(source["records"]),
        )
        for source in source_identities
    )
    audited_source_bindings = Counter(
        (
            int(identity["bound_prompt_source"]["bytes"]),
            str(identity["bound_prompt_source"]["sha256"]),
            int(identity["bound_prompt_source"]["records"]),
        )
        for identity in adapter_provenance_identities
    )
    if audited_source_bindings != declared_source_bindings:
        raise ValueError(
            "adapter provenance must bind every declared prompt source exactly once"
        )
    evidence_identities = [
        _verified_artifact(
            artifact,
            max_bytes=limits["max_license_evidence_bytes"],
            artifact_kind="license evidence",
        )
        for artifact in sorted(evidence, key=lambda artifact: artifact.name)
    ]
    output_payload, dedup_audit = _deduplicated_output(rows)

    if (
        _read_bounded_regular_file(
            contract_file,
            max_bytes=_MAX_CONTRACT_BYTES,
            label="contract",
        )
        != contract_payload
    ):
        raise ValueError("contract changed while the suite was being frozen")

    output_identity = {
        "path": _portable_path(output_file, relative_to=manifest_file.parent),
        "format": "canonical_jsonl",
        "bytes": len(output_payload),
        "sha256": _sha256(output_payload),
        "records": dedup_audit["unique_normalized_prompts"],
        "fields": ["prompt", "source_case_id_sha256"],
    }
    manifest_without_hash = {
        "kind": MANIFEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_prompt_only_pretraining_decontamination_suite",
        "suite": suite,
        "contract": {
            "path": _portable_path(
                contract_file,
                relative_to=manifest_file.parent,
            ),
            "bytes": len(contract_payload),
            "sha256": _sha256(contract_payload),
        },
        "benchmark_plan": benchmark_plan_identity,
        "sources": source_identities,
        "adapter_provenance": adapter_provenance_identities,
        "license_evidence": evidence_identities,
        "raw_artifacts": raw_artifact_identities,
        "limits": {
            **limits,
            "hard_max_benchmark_plan_bytes": _MAX_BENCHMARK_PLAN_BYTES,
            "hard_max_source_artifacts": _MAX_SOURCE_ARTIFACTS,
            "hard_max_adapter_provenance_artifacts": (
                _MAX_ADAPTER_PROVENANCE_ARTIFACTS
            ),
            "hard_max_license_evidence_artifacts": (
                _MAX_LICENSE_EVIDENCE_ARTIFACTS
            ),
            "hard_max_total_source_bytes": _MAX_TOTAL_SOURCE_BYTES,
            "hard_max_total_adapter_provenance_bytes": (
                _MAX_TOTAL_ADAPTER_PROVENANCE_BYTES
            ),
            "hard_max_total_license_evidence_bytes": (
                _MAX_TOTAL_LICENSE_EVIDENCE_BYTES
            ),
            "hard_max_prompt_bytes": _MAX_PROMPT_BYTES,
            "hard_max_source_case_id_bytes": _MAX_SOURCE_CASE_ID_BYTES,
            "hard_max_output_bytes": _MAX_OUTPUT_BYTES,
        },
        "deduplication_audit": dedup_audit,
        "output": output_identity,
        "isolation": {
            "purpose": "pretraining_corpus_decontamination_only",
            "prompt_only": True,
            "contains_labels_or_expected_outputs": False,
            "fresh_labeled_evaluation_evidence": False,
            "benchmark_score_evidence": False,
            "permitted_training_use": (
                "prompt-only denylist may be used only to exclude matching corpus documents"
            ),
            "limitations": (
                "Binds and normalizes only the declared offline adapter exports, adapter-audit "
                "provenance, and license-evidence bytes. It does not independently prove the "
                "semantics of an adapter audit, adjudicate licensing, detect semantic paraphrases, "
                "create a fresh labeled evaluation slice, or support benchmark-score claims."
            ),
        },
    }
    manifest_self_sha256 = _sha256(_canonical_bytes(manifest_without_hash))
    manifest_payload = _canonical_bytes(
        {
            **manifest_without_hash,
            "manifest_self_sha256": manifest_self_sha256,
        }
    )

    _assert_existing_or_absent(output_file, output_payload)
    _assert_existing_or_absent(manifest_file, manifest_payload)
    _publish_atomic(output_file, output_payload)
    _publish_atomic(manifest_file, manifest_payload)
    return json.loads(manifest_payload)


def verify_evaluation_denylist_suite(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Rebuild one frozen suite from its bound contract and verify every artifact.

    Verification is intentionally stricter than checking the manifest and prompt output alone:
    it reruns the deterministic freeze against the original prompt export, source-specific adapter
    audit, and license evidence. Existing artifacts must already be present and byte-identical;
    this function never reconstructs a missing output.
    """

    manifest_file = Path(manifest_path)
    try:
        payload = _read_bounded_regular_file(
            manifest_file,
            max_bytes=_MAX_CONTRACT_BYTES,
            label="suite provenance manifest",
        )
    except ValueError as error:
        raise ValueError(
            "suite provenance manifest must be a bounded regular non-symlink file"
        ) from error
    manifest = _strict_json_loads(payload, label=str(manifest_file))
    if not isinstance(manifest, Mapping):
        raise ValueError("suite provenance manifest must be a JSON object")
    _exact_keys(manifest, _MANIFEST_KEYS, label="suite provenance manifest")
    if (
        manifest.get("kind") != MANIFEST_KIND
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(
            f"suite provenance manifest must be {MANIFEST_KIND!r} "
            f"schema_version {SCHEMA_VERSION}"
        )
    declared_self_hash = manifest.get("manifest_self_sha256")
    if (
        not isinstance(declared_self_hash, str)
        or _SHA256.fullmatch(declared_self_hash) is None
    ):
        raise ValueError("suite provenance manifest self-hash is invalid")
    without_hash = dict(manifest)
    without_hash.pop("manifest_self_sha256")
    if _sha256(_canonical_bytes(without_hash)) != declared_self_hash:
        raise ValueError("suite provenance manifest self-hash mismatch")

    contract = manifest.get("contract")
    output = manifest.get("output")
    if not isinstance(contract, Mapping) or not isinstance(output, Mapping):
        raise ValueError("suite provenance contract and output must be objects")
    _exact_keys(contract, _MANIFEST_CONTRACT_KEYS, label="suite provenance contract")
    _exact_keys(output, _MANIFEST_OUTPUT_KEYS, label="suite provenance output")
    raw_contract_path = _nonempty_string(
        contract.get("path"),
        label="suite provenance contract.path",
    )
    raw_output_path = _nonempty_string(
        output.get("path"),
        label="suite provenance output.path",
    )
    if Path(raw_contract_path).is_absolute() or Path(raw_output_path).is_absolute():
        raise ValueError("suite provenance paths must be portable relative paths")
    contract_file = manifest_file.parent / raw_contract_path
    output_file = manifest_file.parent / raw_output_path
    if not output_file.is_file() or output_file.is_symlink():
        raise ValueError("suite provenance prompt output is missing or not a regular file")

    rebuilt = freeze_evaluation_denylist_suite(
        contract_file,
        output_path=output_file,
        manifest_path=manifest_file,
    )
    if rebuilt != manifest:
        raise RuntimeError("rebuilt suite provenance differs from the frozen manifest")
    return rebuilt
