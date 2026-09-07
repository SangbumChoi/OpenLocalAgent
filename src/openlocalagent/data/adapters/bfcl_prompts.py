"""Strict offline BFCL-v4 prompt extraction for corpus decontamination.

This module is deliberately not a BFCL scorer and does not read possible answers.  It accepts
only caller-declared, content-addressed BFCL-v4 input JSONL files, validates their complete row
shape, and emits prompt-only rows for the generic evaluation denylist freezer.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

BFCL_SOURCE_MANIFEST_KIND = "localagent_bfcl_v4_prompt_source_manifest"
BFCL_SOURCE_MANIFEST_SCHEMA_VERSION = 1
BFCL_PROMPT_AUDIT_KIND = "localagent_bfcl_v4_prompt_export_audit"
BFCL_PROMPT_AUDIT_SCHEMA_VERSION = 1
BFCL_PROMPT_ADAPTER = "bfcl-v4-prompt-rows-v1"
PRODUCTION_BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
PRODUCTION_BFCL_CATEGORIES = frozenset(
    {"simple_python", "multiple", "parallel", "parallel_multiple"}
)
PRODUCTION_BFCL_SOURCE_IDENTITIES: Mapping[str, tuple[int, int, str]] = (
    MappingProxyType(
        {
            "simple_python": (
                283_274,
                400,
                "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
            ),
            "multiple": (
                316_583,
                200,
                "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
            ),
            "parallel": (
                171_896,
                200,
                "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
            ),
            "parallel_multiple": (
                347_080,
                200,
                "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
            ),
        }
    )
)

DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024
DEFAULT_MAX_SOURCE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_SOURCE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_LINE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_SOURCE_ROWS = 50_000
DEFAULT_MAX_OUTPUT_ROWS = 250_000
DEFAULT_MAX_SOURCES = 128

_CATEGORY_RE = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
_SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_ROW_KEYS = frozenset({"id", "question", "function"})
_MESSAGE_KEYS = frozenset({"role", "content"})
_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})


@dataclass(frozen=True)
class BFCLPromptLimits:
    """Resource limits for bounded local BFCL parsing."""

    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
    max_total_source_bytes: int = DEFAULT_MAX_TOTAL_SOURCE_BYTES
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    max_source_rows: int = DEFAULT_MAX_SOURCE_ROWS
    max_output_rows: int = DEFAULT_MAX_OUTPUT_ROWS
    max_sources: int = DEFAULT_MAX_SOURCES

    def validate(self) -> None:
        for name, value in (
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_source_bytes", self.max_source_bytes),
            ("max_total_source_bytes", self.max_total_source_bytes),
            ("max_line_bytes", self.max_line_bytes),
            ("max_source_rows", self.max_source_rows),
            ("max_output_rows", self.max_output_rows),
            ("max_sources", self.max_sources),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _Source:
    category: str
    path: Path
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class _PromptRow:
    sort_key: tuple[str, str, str, int, int, int]
    source_case_id: str
    prompt: str
    component: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bounded_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit: {path}")
    return payload


def _matches_payload(path: Path, payload: bytes) -> bool:
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


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON") from error


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} schema drift: missing={missing}, extra={extra}")


def _require_file(path: Path, *, label: str) -> int:
    if not path.is_file():
        raise ValueError(f"{label} is missing or is not a file: {path}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    return path.stat().st_size


def _validate_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _parse_source_manifest(
    manifest_path: Path,
    *,
    limits: BFCLPromptLimits,
) -> tuple[bytes, str, list[_Source]]:
    manifest_size = _require_file(manifest_path, label="BFCL source manifest")
    if manifest_size > limits.max_manifest_bytes:
        raise ValueError(
            "BFCL source manifest exceeds "
            f"the {limits.max_manifest_bytes}-byte limit: {manifest_path}"
        )
    manifest_payload = _read_bounded_file(
        manifest_path,
        max_bytes=limits.max_manifest_bytes,
        label="BFCL source manifest",
    )
    try:
        manifest_text = manifest_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("BFCL source manifest is not UTF-8") from error
    value = _strict_json_loads(manifest_text, label="BFCL source manifest")
    if not isinstance(value, dict):
        raise ValueError("BFCL source manifest must be an object")
    _require_exact_keys(
        value,
        frozenset({"kind", "schema_version", "benchmark", "revision", "sources"}),
        label="BFCL source manifest",
    )
    if value["kind"] != BFCL_SOURCE_MANIFEST_KIND:
        raise ValueError("BFCL source manifest kind is unsupported")
    if value["schema_version"] != BFCL_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("BFCL source manifest schema_version is unsupported")
    if value["benchmark"] != "bfcl-v4":
        raise ValueError("BFCL source manifest benchmark must be 'bfcl-v4'")
    revision = value["revision"]
    if not isinstance(revision, str) or _GIT_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("BFCL source manifest revision must be a lowercase 40-hex Git commit")
    raw_sources = value["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("BFCL source manifest sources must be a non-empty list")
    if len(raw_sources) > limits.max_sources:
        raise ValueError(
            f"BFCL source manifest exceeds the {limits.max_sources}-source limit"
        )

    sources: list[_Source] = []
    categories: set[str] = set()
    resolved_paths: set[Path] = set()
    for index, raw in enumerate(raw_sources):
        label = f"BFCL source manifest sources[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(
            raw,
            frozenset({"category", "path", "bytes", "sha256"}),
            label=label,
        )
        category = raw["category"]
        if not isinstance(category, str) or _CATEGORY_RE.fullmatch(category) is None:
            raise ValueError(f"{label}.category is invalid")
        if category in categories:
            raise ValueError(f"duplicate BFCL category {category!r}")
        categories.add(category)

        raw_path = raw["path"]
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{label}.path must be a non-empty string")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if any(part.casefold() == "possible_answer" for part in path.parts):
            raise ValueError("BFCL possible_answer/gold paths are forbidden")
        expected_name = f"BFCL_v4_{category}.json"
        if path.name != expected_name:
            raise ValueError(
                f"{label}.path must name the declared v4 category file {expected_name!r}"
            )
        resolved = path.resolve()
        if resolved in resolved_paths:
            raise ValueError("the same BFCL source file was declared more than once")
        resolved_paths.add(resolved)

        expected_bytes = _validate_nonnegative_int(raw["bytes"], label=f"{label}.bytes")
        expected_sha256 = raw["sha256"]
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError(f"{label}.sha256 must be lowercase 64-hex")
        sources.append(
            _Source(
                category=category,
                path=path,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
        )
    return manifest_payload, revision, sources


def _validate_json_tree(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string object key")
            _validate_json_tree(item, label=f"{label}.{key}")
        return
    raise ValueError(f"{label} contains a non-JSON value")


def _parse_question(
    value: Any,
    *,
    category: str,
    source_id: str,
    file_name: str,
) -> list[_PromptRow]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"BFCL case {source_id!r} question must be a non-empty turn list")
    rows: list[_PromptRow] = []
    for turn_index, turn in enumerate(value):
        if not isinstance(turn, list) or not turn:
            raise ValueError(
                f"BFCL case {source_id!r} question[{turn_index}] "
                "must be a non-empty message list"
            )
        for message_index, message in enumerate(turn):
            label = (
                f"BFCL case {source_id!r} "
                f"question[{turn_index}][{message_index}]"
            )
            if not isinstance(message, dict):
                raise ValueError(f"{label} must be an object")
            _require_exact_keys(message, _MESSAGE_KEYS, label=label)
            role = message["role"]
            content = message["content"]
            if not isinstance(role, str) or not role:
                raise ValueError(f"{label}.role must be a non-empty string")
            if not isinstance(content, str) or not content:
                raise ValueError(f"{label}.content must be a non-empty string")
            rows.append(
                _PromptRow(
                    sort_key=(
                        category,
                        file_name,
                        source_id,
                        0,
                        turn_index,
                        message_index,
                    ),
                    source_case_id=(
                        f"bfcl:{category}:{source_id}:"
                        f"question:{turn_index:04d}:{message_index:04d}"
                    ),
                    prompt=content,
                    component="question",
                )
            )
    return rows


def _parse_functions(
    value: Any,
    *,
    category: str,
    source_id: str,
    file_name: str,
) -> list[_PromptRow]:
    if not isinstance(value, list):
        raise ValueError(f"BFCL case {source_id!r} function must be a list")
    rows: list[_PromptRow] = []
    for function_index, function in enumerate(value):
        label = f"BFCL case {source_id!r} function[{function_index}]"
        if not isinstance(function, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(function, _FUNCTION_KEYS, label=label)
        name = function["name"]
        description = function["description"]
        parameters = function["parameters"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.name must be a non-empty string")
        if not isinstance(description, str):
            raise ValueError(f"{label}.description must be a string")
        if not isinstance(parameters, dict):
            raise ValueError(f"{label}.parameters must be an object")
        _validate_json_tree(function, label=label)
        canonical = _canonical_json_bytes(function).decode("utf-8")
        rows.append(
            _PromptRow(
                sort_key=(
                    category,
                    file_name,
                    source_id,
                    1,
                    function_index,
                    0,
                ),
                source_case_id=(
                    f"bfcl:{category}:{source_id}:function:{function_index:04d}"
                ),
                prompt=canonical,
                component="function",
            )
        )
    return rows


def _parse_source_rows(
    source: _Source,
    *,
    limits: BFCLPromptLimits,
    seen_ids: set[str],
) -> tuple[list[_PromptRow], int, str]:
    source_size = _require_file(source.path, label=f"BFCL category {source.category!r}")
    if source_size > limits.max_source_bytes:
        raise ValueError(
            f"BFCL category {source.category!r} exceeds "
            f"the {limits.max_source_bytes}-byte source limit"
        )
    with source.path.open("rb") as source_handle:
        source_payload = source_handle.read(limits.max_source_bytes + 1)
    if len(source_payload) > limits.max_source_bytes:
        raise ValueError(
            f"BFCL category {source.category!r} exceeds "
            f"the {limits.max_source_bytes}-byte source limit"
        )
    if len(source_payload) != source.expected_bytes:
        raise ValueError(
            f"BFCL category {source.category!r} byte-size mismatch: "
            f"expected {source.expected_bytes}, got {len(source_payload)}"
        )
    observed_sha256 = _sha256(source_payload)
    if observed_sha256 != source.expected_sha256:
        raise ValueError(f"BFCL category {source.category!r} SHA-256 mismatch")

    rows: list[_PromptRow] = []
    source_rows = 0
    with io.BytesIO(source_payload) as handle:
        while True:
            raw_line = handle.readline(limits.max_line_bytes + 1)
            if not raw_line:
                break
            source_rows += 1
            if source_rows > limits.max_source_rows:
                raise ValueError(
                    f"BFCL category {source.category!r} exceeds "
                    f"the {limits.max_source_rows}-row limit"
                )
            if len(raw_line) > limits.max_line_bytes:
                raise ValueError(
                    f"BFCL category {source.category!r} line {source_rows} exceeds "
                    f"the {limits.max_line_bytes}-byte line limit"
                )
            if not raw_line.strip():
                raise ValueError(
                    f"BFCL category {source.category!r} line {source_rows} is blank"
                )
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"BFCL category {source.category!r} line {source_rows} is not UTF-8"
                ) from error
            record = _strict_json_loads(
                decoded,
                label=f"BFCL category {source.category!r} line {source_rows}",
            )
            if not isinstance(record, dict):
                raise ValueError(
                    f"BFCL category {source.category!r} line {source_rows} "
                    "must be an object"
                )
            _require_exact_keys(
                record,
                _ROW_KEYS,
                label=f"BFCL category {source.category!r} line {source_rows}",
            )
            source_id = record["id"]
            if (
                not isinstance(source_id, str)
                or len(source_id) > 512
                or _SOURCE_ID_RE.fullmatch(source_id) is None
            ):
                raise ValueError(
                    f"BFCL category {source.category!r} line {source_rows} "
                    "has an invalid id"
                )
            if source_id in seen_ids:
                raise ValueError(f"duplicate BFCL id {source_id!r}")
            seen_ids.add(source_id)
            rows.extend(
                _parse_question(
                    record["question"],
                    category=source.category,
                    source_id=source_id,
                    file_name=source.path.name,
                )
            )
            rows.extend(
                _parse_functions(
                    record["function"],
                    category=source.category,
                    source_id=source_id,
                    file_name=source.path.name,
                )
            )
            if len(rows) > limits.max_output_rows:
                raise ValueError(
                    f"BFCL category {source.category!r} exceeds "
                    f"the {limits.max_output_rows}-prompt-row limit"
                )
    if source_rows == 0:
        raise ValueError(f"BFCL category {source.category!r} is empty")
    return rows, source_rows, observed_sha256


def _assert_existing_exact(path: Path, payload: bytes) -> None:
    if path.exists() and not _matches_payload(path, payload):
        raise RuntimeError(f"refusing to overwrite drifted derived artifact: {path}")


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        if not _matches_payload(path, payload):
            raise RuntimeError(f"refusing to overwrite drifted derived artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if not _matches_payload(path, payload):
                raise RuntimeError(
                    f"refusing to overwrite concurrently-created artifact: {path}"
                )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    if not _matches_payload(path, payload):
        raise RuntimeError(f"published artifact failed verification: {path}")


def _rows_payload(rows: Iterable[_PromptRow]) -> bytes:
    return b"".join(
        _canonical_json_bytes(
            {"source_case_id": row.source_case_id, "prompt": row.prompt},
            newline=True,
        )
        for row in rows
    )


def export_bfcl_prompt_rows(
    source_manifest_path: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
    *,
    limits: BFCLPromptLimits = BFCLPromptLimits(),
) -> dict[str, Any]:
    """Validate identity-bound BFCL-v4 input JSONL and publish prompt-only rows.

    The manifest is the caller's explicit category declaration.  Its schema is::

        {
          "kind": "localagent_bfcl_v4_prompt_source_manifest",
          "schema_version": 1,
          "benchmark": "bfcl-v4",
          "revision": "<lowercase 40-hex commit>",
          "sources": [
            {"category": "simple_python", "path": "...", "bytes": 1,
             "sha256": "<lowercase 64-hex>"}
          ]
        }

    Existing destinations are accepted only when they reproduce byte-for-byte.
    """

    limits.validate()
    manifest_path = Path(source_manifest_path)
    output = Path(output_path)
    audit_file = Path(audit_path)
    if output.resolve() == audit_file.resolve():
        raise ValueError("output_path and audit_path must be different files")

    manifest_payload, revision, sources = _parse_source_manifest(
        manifest_path,
        limits=limits,
    )
    protected = {manifest_path.resolve(), *(source.path.resolve() for source in sources)}
    if output.resolve() in protected or audit_file.resolve() in protected:
        raise ValueError("derived outputs must not overwrite BFCL source artifacts")

    total_source_bytes = sum(source.expected_bytes for source in sources)
    if total_source_bytes > limits.max_total_source_bytes:
        raise ValueError(
            "declared BFCL sources exceed "
            f"the {limits.max_total_source_bytes}-byte total source limit"
        )

    prompt_rows: list[_PromptRow] = []
    source_audit: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in sorted(sources, key=lambda item: (item.category, item.path.name)):
        parsed, source_rows, observed_sha256 = _parse_source_rows(
            source,
            limits=limits,
            seen_ids=seen_ids,
        )
        prompt_rows.extend(parsed)
        if len(prompt_rows) > limits.max_output_rows:
            raise ValueError(
                f"BFCL export exceeds the {limits.max_output_rows}-prompt-row limit"
            )
        source_audit.append(
            {
                "bytes": source.expected_bytes,
                "category": source.category,
                "file": source.path.name,
                "rows": source_rows,
                "sha256": observed_sha256,
            }
        )

    prompt_rows.sort(key=lambda row: row.sort_key)
    source_case_ids = [row.source_case_id for row in prompt_rows]
    if len(source_case_ids) != len(set(source_case_ids)):
        raise ValueError("derived BFCL prompt source_case_id collision")
    output_payload = _rows_payload(prompt_rows)
    question_rows = sum(row.component == "question" for row in prompt_rows)
    function_rows = sum(row.component == "function" for row in prompt_rows)
    declared_categories = [item["category"] for item in source_audit]
    split = "+".join(sorted(declared_categories))
    production_shape = (
        revision == PRODUCTION_BFCL_REVISION
        and set(declared_categories) == PRODUCTION_BFCL_CATEGORIES
    )
    if production_shape:
        observed_identities = {
            str(item["category"]): (
                int(item["bytes"]),
                int(item["rows"]),
                str(item["sha256"]),
            )
            for item in source_audit
        }
        if observed_identities != dict(PRODUCTION_BFCL_SOURCE_IDENTITIES):
            raise ValueError(
                "BFCL production sources do not match the authoritative pinned "
                "byte/row/SHA-256 identities"
            )
    mode = "production" if production_shape else "fixture"
    audit: dict[str, Any] = {
        "kind": BFCL_PROMPT_AUDIT_KIND,
        "schema_version": BFCL_PROMPT_AUDIT_SCHEMA_VERSION,
        "adapter": BFCL_PROMPT_ADAPTER,
        "freeze_binding": {
            "adapter": BFCL_PROMPT_ADAPTER,
            "benchmark": "bfcl-v4",
            "mode": mode,
            "revision": revision,
            "split": split,
            "prompt_only": True,
            "contains_current_step_labels": False,
            "output": {
                "bytes": len(output_payload),
                "sha256": _sha256(output_payload),
                "records": len(prompt_rows),
            },
        },
        "purpose": "prompt_only_corpus_decontamination",
        "benchmark": "bfcl-v4",
        "mode": mode,
        "revision": revision,
        "split": split,
        "source_manifest": {
            "bytes": len(manifest_payload),
            "file": manifest_path.name,
            "sha256": _sha256(manifest_payload),
        },
        "sources": source_audit,
        "selection": {
            "caller_declared_categories": declared_categories,
            "source_rows": sum(int(item["rows"]) for item in source_audit),
            "question_prompt_rows": question_rows,
            "function_spec_prompt_rows": function_rows,
            "ordering": "category,file,id,question-before-function,component-index",
        },
        "output": {
            "bytes": len(output_payload),
            "file": output.name,
            "rows": len(prompt_rows),
            "sha256": _sha256(output_payload),
            "row_keys": ["prompt", "source_case_id"],
        },
        "limits": {
            "max_manifest_bytes": limits.max_manifest_bytes,
            "max_line_bytes": limits.max_line_bytes,
            "max_output_rows": limits.max_output_rows,
            "max_source_bytes": limits.max_source_bytes,
            "max_source_rows": limits.max_source_rows,
            "max_sources": limits.max_sources,
            "max_total_source_bytes": limits.max_total_source_bytes,
        },
        "boundary": (
            "decontamination prompts only; no possible answers, official scoring, "
            "or chronologically fresh-evaluation claim"
        ),
    }
    audit_payload = _canonical_json_bytes(audit, newline=True)

    _assert_existing_exact(output, output_payload)
    _assert_existing_exact(audit_file, audit_payload)
    _publish_atomic(output, output_payload)
    _publish_atomic(audit_file, audit_payload)
    return audit


__all__ = [
    "BFCLPromptLimits",
    "BFCL_PROMPT_ADAPTER",
    "BFCL_PROMPT_AUDIT_KIND",
    "BFCL_SOURCE_MANIFEST_KIND",
    "PRODUCTION_BFCL_CATEGORIES",
    "PRODUCTION_BFCL_REVISION",
    "PRODUCTION_BFCL_SOURCE_IDENTITIES",
    "export_bfcl_prompt_rows",
]
