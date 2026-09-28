"""Fail-closed validation for the public realistic-agent evaluation matrix.

The matrix is deliberately separate from the executable training catalog.  It records public
source links and the *kind* of evidence a WebGPU text-first model can obtain, but it never turns an
evaluation benchmark into training data implicitly.  Payloads, credentials, emulator images, and
benchmark task text stay outside Git unless a source-specific adapter and split receipt authorize
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MATRIX_KIND = "localagent_public_realistic_eval_matrix"
MATRIX_SCHEMA_VERSION = 1
_FAMILIES = frozenset({"mobile", "browser", "computer", "tool_api", "terminal"})
_ACCESS = frozenset({"public_download", "public_runtime", "protected", "terms_review"})
_POLICIES = frozenset({"train", "eval_only", "no_static_data", "restricted"})
_STATUSES = frozenset(
    {
        "measured",
        "measured_official_text_projection",
        "metadata_only",
        "manifest_audited_native_pending",
        "native_mock_diagnostic",
        "adapter_ready",
        "runtime_pending",
        "not_started",
    }
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _https(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if not text.startswith("https://"):
        raise ValueError(f"{label} must use https://")
    return text


def validate_matrix(raw: object) -> dict[str, Any]:
    """Validate and detach a public matrix, rejecting ambiguous train/eval rows."""

    if not isinstance(raw, Mapping):
        raise ValueError("matrix must be a mapping")
    matrix = dict(raw)
    if matrix.get("kind") != MATRIX_KIND:
        raise ValueError(f"matrix.kind must be {MATRIX_KIND!r}")
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError(f"matrix.schema_version must be {MATRIX_SCHEMA_VERSION}")
    entries = matrix.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("matrix.entries must be a non-empty list")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError(f"entries[{index}] must be a mapping")
        row = dict(item)
        prefix = f"entries[{index}]"
        entry_id = _text(row.get("id"), label=f"{prefix}.id")
        if entry_id in seen:
            raise ValueError(f"duplicate matrix id: {entry_id}")
        seen.add(entry_id)
        family = _text(row.get("family"), label=f"{prefix}.family")
        if family not in _FAMILIES:
            raise ValueError(f"{prefix}.family has unsupported value {family!r}")
        _https(row.get("source_url"), label=f"{prefix}.source_url")
        _https(row.get("paper_url"), label=f"{prefix}.paper_url")
        access = _text(row.get("access_status"), label=f"{prefix}.access_status")
        if access not in _ACCESS:
            raise ValueError(f"{prefix}.access_status has unsupported value {access!r}")
        policy = _text(row.get("train_policy"), label=f"{prefix}.train_policy")
        if policy not in _POLICIES:
            raise ValueError(f"{prefix}.train_policy has unsupported value {policy!r}")
        license_name = _text(row.get("license"), label=f"{prefix}.license")
        if policy == "train" and access != "public_download":
            raise ValueError(f"{prefix} train rows require public_download access")
        if policy == "train" and license_name.lower() in {"unknown", "unverified", "terms_review"}:
            raise ValueError(f"{prefix} train rows require a reviewed license")
        for key in ("modalities", "webgpu_projection", "primary_metric", "split_rule", "notes"):
            value = row.get(key)
            if isinstance(value, list):
                if not value or not all(isinstance(item, str) and item.strip() for item in value):
                    raise ValueError(f"{prefix}.{key} must be a non-empty string list")
            else:
                _text(value, label=f"{prefix}.{key}")
        status = _text(row.get("local_status"), label=f"{prefix}.local_status")
        if status not in _STATUSES:
            raise ValueError(f"{prefix}.local_status has unsupported value {status!r}")
        if "source_revision" in row and row["source_revision"] is not None:
            _text(row["source_revision"], label=f"{prefix}.source_revision")
        normalized.append(row)

    matrix["entries"] = normalized
    return matrix


def load_matrix(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON matrix."""

    matrix_path = Path(path)
    try:
        raw = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read matrix {matrix_path}: {error}") from error
    return validate_matrix(raw)


def entries_by_family(matrix: Mapping[str, Any], family: str) -> tuple[Mapping[str, Any], ...]:
    """Return validated entries for one modality family."""

    validated = validate_matrix(matrix)
    return tuple(row for row in validated["entries"] if row["family"] == family)


def trainable_entries(matrix: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return rows that a source-specific acquisition may propose for training."""

    validated = validate_matrix(matrix)
    return tuple(row for row in validated["entries"] if row["train_policy"] == "train")
