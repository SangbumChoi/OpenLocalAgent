"""Validation and provenance helpers for realistic agent data and benchmarks.

The catalog is intentionally descriptive rather than an implicit downloader.  A row can describe
an executable environment (for example AndroidWorld or MCPMark) even when it has no static
training file.  ``train_policy`` is the safety boundary used by acquisition and training code:
only rows explicitly marked ``train`` may become training input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

CATALOG_KIND = "localagent_realistic_agent_catalog"
CATALOG_SCHEMA_VERSION = 1
_FAMILIES = frozenset({"mobile", "computer", "browser", "tool_api", "terminal"})
_ACCESS = frozenset({"public_download", "public_runtime", "protected", "terms_review"})
_TRAIN_POLICIES = frozenset({"train", "eval_only", "restricted", "no_static_data"})
_SCALE_KINDS = frozenset({"dataset", "environment", "benchmark", "mixture"})
_SHA256 = frozenset("0123456789abcdef")


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_https(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if not text.startswith("https://"):
        raise ValueError(f"{label} must use https://")
    return text


def _require_string_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = [_require_text(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _require_nonnegative_counts(value: Mapping[str, Any], *, label: str) -> None:
    for key, count in value.items():
        if key == "kind":
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def validate_catalog(raw: object) -> dict[str, Any]:
    """Validate and return a detached catalog mapping.

    The function rejects ambiguous rows rather than trying to infer license or contamination
    policy.  This keeps accidental benchmark-to-training leakage fail-closed.
    """

    root = dict(_require_mapping(raw, label="catalog"))
    if root.get("kind") != CATALOG_KIND:
        raise ValueError(f"catalog.kind must be {CATALOG_KIND!r}")
    if root.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"catalog.schema_version must be {CATALOG_SCHEMA_VERSION}")
    entries = root.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog.entries must be a non-empty list")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        row = dict(_require_mapping(item, label=f"entries[{index}]"))
        entry_id = _require_text(row.get("id"), label=f"entries[{index}].id")
        if entry_id in seen:
            raise ValueError(f"duplicate catalog id: {entry_id}")
        seen.add(entry_id)
        family = _require_text(row.get("family"), label=f"entries[{index}].family")
        if family not in _FAMILIES:
            raise ValueError(f"entries[{index}].family has unsupported value {family!r}")
        access = _require_text(row.get("access_status"), label=f"entries[{index}].access_status")
        if access not in _ACCESS:
            raise ValueError(f"entries[{index}].access_status has unsupported value {access!r}")
        policy = _require_text(row.get("train_policy"), label=f"entries[{index}].train_policy")
        if policy not in _TRAIN_POLICIES:
            raise ValueError(f"entries[{index}].train_policy has unsupported value {policy!r}")
        if policy == "train" and access != "public_download":
            raise ValueError(f"entries[{index}] train policy requires public_download access")

        for key in ("name", "source_url", "paper_url", "code_url", "notes"):
            if key not in row:
                raise ValueError(f"entries[{index}] is missing {key!r}")
        _require_text(row["name"], label=f"entries[{index}].name")
        for key in ("source_url", "paper_url", "code_url"):
            _require_https(row[key], label=f"entries[{index}].{key}")

        license_info = _require_mapping(row.get("license"), label=f"entries[{index}].license")
        license_name = _require_text(
            license_info.get("name"), label=f"entries[{index}].license.name"
        )
        if license_name.lower() in {"unknown", "unverified"} and policy == "train":
            raise ValueError(f"entries[{index}] train rows require a verified license")
        if "url" in license_info:
            _require_https(license_info["url"], label=f"entries[{index}].license.url")
        if "evidence" not in license_info:
            raise ValueError(f"entries[{index}].license.evidence is required")
        _require_https(license_info["evidence"], label=f"entries[{index}].license.evidence")

        scale = _require_mapping(row.get("scale"), label=f"entries[{index}].scale")
        scale_kind = _require_text(scale.get("kind"), label=f"entries[{index}].scale.kind")
        if scale_kind not in _SCALE_KINDS:
            raise ValueError(f"entries[{index}].scale.kind has unsupported value {scale_kind!r}")
        _require_nonnegative_counts(scale, label=f"entries[{index}].scale")

        observation = _require_mapping(
            row.get("observation"), label=f"entries[{index}].observation"
        )
        action = _require_mapping(row.get("action"), label=f"entries[{index}].action")
        for section, section_value in (("observation", observation), ("action", action)):
            _require_string_list(
                section_value.get("modalities"), label=f"entries[{index}].{section}.modalities"
            )
            _require_string_list(
                section_value.get("formats"), label=f"entries[{index}].{section}.formats"
            )
        _require_string_list(row.get("domains"), label=f"entries[{index}].domains")
        _require_mapping(row.get("integration"), label=f"entries[{index}].integration")
        projection = _require_mapping(
            row.get("webgpu_projection"), label=f"entries[{index}].webgpu_projection"
        )
        _require_text(projection.get("status"), label=f"entries[{index}].webgpu_projection.status")
        _require_text(
            projection.get("recommended_observation"),
            label=f"entries[{index}].webgpu_projection.recommended_observation",
        )
        contamination = _require_mapping(
            row.get("contamination"), label=f"entries[{index}].contamination"
        )
        _require_text(contamination.get("split"), label=f"entries[{index}].contamination.split")
        _require_text(
            contamination.get("rule"), label=f"entries[{index}].contamination.rule"
        )

        normalized.append(row)

    root["entries"] = normalized
    return root


def load_catalog(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load, validate, and fingerprint a catalog file."""

    catalog_path = Path(path)
    try:
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read catalog {catalog_path}: {error}") from error
    catalog = validate_catalog(raw)
    return catalog, hashlib.sha256(_canonical_json(catalog)).hexdigest()


def train_entries(catalog: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return only rows explicitly authorized as trainable."""

    validated = validate_catalog(catalog)
    return tuple(row for row in validated["entries"] if row["train_policy"] == "train")


def eval_entries(catalog: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return benchmark/runtime rows that must stay out of training."""

    validated = validate_catalog(catalog)
    return tuple(row for row in validated["entries"] if row["train_policy"] != "train")
