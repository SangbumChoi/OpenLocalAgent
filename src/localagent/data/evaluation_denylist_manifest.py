"""Compose and verify provenance-bound evaluation denylist list manifests.

The per-suite freezer publishes a prompt-only JSONL file and a self-hashed provenance
manifest.  This module binds a set of those pairs into the list-manifest format consumed by
``scripts/prepare_corpus.py``.  Suite names are taken only from verified provenance, paths are
portable relative paths, and both producer and consumer use the same strict verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localagent.data.evaluation_denylist_suite import (
    MANIFEST_KIND as SUITE_PROVENANCE_KIND,
)
from localagent.data.evaluation_denylist_suite import (
    SCHEMA_VERSION as SUITE_PROVENANCE_SCHEMA_VERSION,
)
from localagent.data.evaluation_denylist_suite import (
    verify_evaluation_denylist_suite,
)

MANIFEST_KIND = "localagent_evaluation_denylist_manifest"
SCHEMA_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUITE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

_MAX_LIST_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_SUITE_PROVENANCE_BYTES = 4 * 1024 * 1024
_MAX_SUITE_CONTRACT_BYTES = 1024 * 1024
_MAX_TOTAL_SUITE_PROVENANCE_BYTES = 64 * 1024 * 1024
_MAX_SUITES = 128
_MAX_SUITE_NAME_BYTES = 256
_MAX_PATH_BYTES = 4096
_MAX_SUITE_OUTPUT_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_OUTPUT_BYTES = 512 * 1024 * 1024
_MAX_SUITE_OUTPUT_RECORDS = 250_000
_MAX_TOTAL_OUTPUT_RECORDS = 1_000_000
_MAX_OUTPUT_RECORD_BYTES = 4 * 1024 * 1024
_MAX_SUITE_ARTIFACTS = 32

_LIST_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "required_suites",
        "suites",
        "manifest_self_sha256",
    }
)
_LIST_SUITE_KEYS = frozenset(
    {"name", "path", "bytes", "sha256", "provenance"}
)
_LIST_PROVENANCE_KEYS = frozenset(
    {"path", "bytes", "sha256", "manifest_self_sha256"}
)
_SUITE_PROVENANCE_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "status",
        "suite",
        "contract",
        "benchmark_plan",
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
_SUITE_KEYS = frozenset({"name", "benchmark", "revision", "split", "adapter"})
_ADAPTER_KEYS = frozenset({"name", "version"})
_CONTRACT_IDENTITY_KEYS = frozenset({"path", "bytes", "sha256"})
_BENCHMARK_PLAN_IDENTITY_KEYS = frozenset(
    {
        "name",
        "bytes",
        "sha256",
        "plan_kind",
        "plan_schema_version",
        "suite_entry_sha256",
    }
)
_SOURCE_IDENTITY_KEYS = frozenset({"name", "bytes", "sha256", "records"})
_ARTIFACT_IDENTITY_KEYS = frozenset({"name", "bytes", "sha256"})
_RAW_ARTIFACT_IDENTITY_KEYS = frozenset({"name", "bytes", "sha256", "role"})
_ADAPTER_PROVENANCE_IDENTITY_KEYS = frozenset(
    {
        "name",
        "bytes",
        "sha256",
        "adapter",
        "audit_kind",
        "audit_schema_version",
        "bound_prompt_source",
    }
)
_BOUND_PROMPT_SOURCE_KEYS = frozenset({"bytes", "records", "sha256"})
_LIMIT_KEYS = frozenset(
    {
        "max_source_bytes",
        "max_benchmark_plan_bytes",
        "max_adapter_provenance_bytes",
        "max_license_evidence_bytes",
        "max_rows",
        "max_record_bytes",
        "hard_max_source_artifacts",
        "hard_max_benchmark_plan_bytes",
        "hard_max_adapter_provenance_artifacts",
        "hard_max_license_evidence_artifacts",
        "hard_max_total_source_bytes",
        "hard_max_total_adapter_provenance_bytes",
        "hard_max_total_license_evidence_bytes",
        "hard_max_prompt_bytes",
        "hard_max_source_case_id_bytes",
        "hard_max_output_bytes",
    }
)
_DEDUPLICATION_KEYS = frozenset(
    {
        "method",
        "input_rows",
        "unique_normalized_prompts",
        "normalized_prompt_duplicates_removed",
        "normalized_prompt_set_sha256",
        "input_source_case_id_hashes_sha256",
        "representative",
    }
)
_OUTPUT_KEYS = frozenset(
    {"path", "format", "bytes", "sha256", "records", "fields"}
)
_ISOLATION_KEYS = frozenset(
    {
        "purpose",
        "prompt_only",
        "contains_labels_or_expected_outputs",
        "fresh_labeled_evaluation_evidence",
        "benchmark_score_evidence",
        "permitted_training_use",
        "limitations",
    }
)
_OUTPUT_ROW_KEYS = frozenset({"prompt", "source_case_id_sha256"})

_SUITE_STATUS = "frozen_prompt_only_pretraining_decontamination_suite"
_ISOLATION_PURPOSE = "pretraining_corpus_decontamination_only"
_PERMITTED_TRAINING_USE = (
    "prompt-only denylist may be used only to exclude matching corpus documents"
)
_DEDUPLICATION_METHOD = "unicode_nfkc_casefold_token_normalization_v1"
_DEDUPLICATION_REPRESENTATIVE = (
    "lexicographically smallest exact prompt then hashed source ID "
    "within each normalized-prompt group"
)


@dataclass(frozen=True)
class _VerifiedSuite:
    """One suite provenance file and its reverified prompt-only output."""

    name: str
    provenance_path: Path
    provenance_bytes: bytes
    provenance_sha256: str
    manifest_self_sha256: str
    benchmark_plan_bytes: int
    benchmark_plan_sha256: str
    output_path: Path
    output_bytes: int
    output_sha256: str
    output_records: int


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


def _read_json_object(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"{label} is missing or is not a regular non-symlink file: {path}"
        )
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{label} exceeds hard byte cap {max_bytes}: {path}")
    value = _strict_json_loads(raw, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


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


def _nonempty_string(
    value: Any,
    *,
    label: str,
    max_bytes: int | None = None,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty, trimmed string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must contain valid Unicode scalar values") from error
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds hard byte cap {max_bytes}")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _lowercase_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    raw = _nonempty_string(value, label=label, max_bytes=_MAX_PATH_BYTES)
    path = Path(raw)
    if (
        path.is_absolute()
        or "\\" in raw
        or raw.startswith("//")
        or re.match(r"^[A-Za-z]:", raw) is not None
    ):
        raise ValueError(f"{label} must be a portable relative POSIX path")
    return raw


def _portable_path(path: Path, *, relative_to: Path) -> str:
    return Path(
        os.path.relpath(path.resolve(), start=relative_to.resolve())
    ).as_posix()


def _identity(
    raw: Any,
    *,
    label: str,
    expected_keys: frozenset[str],
    named: bool = False,
    records: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    _exact_keys(raw, expected_keys, label=label)
    result: dict[str, Any] = {}
    if named:
        result["name"] = _nonempty_string(
            raw.get("name"),
            label=f"{label}.name",
            max_bytes=_MAX_SUITE_NAME_BYTES,
        )
    result["bytes"] = _positive_int(raw.get("bytes"), label=f"{label}.bytes")
    result["sha256"] = _lowercase_sha256(
        raw.get("sha256"),
        label=f"{label}.sha256",
    )
    if records:
        result["records"] = _positive_int(
            raw.get("records"),
            label=f"{label}.records",
        )
    return result


def _named_identity_array(
    raw: Any,
    *,
    label: str,
    expected_keys: frozenset[str],
    records: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a non-empty array")
    if len(raw) > _MAX_SUITE_ARTIFACTS:
        raise ValueError(f"{label} exceeds hard artifact cap {_MAX_SUITE_ARTIFACTS}")
    identities = [
        _identity(
            value,
            label=f"{label}[{index}]",
            expected_keys=expected_keys,
            named=True,
            records=records,
        )
        for index, value in enumerate(raw)
    ]
    names = [str(identity["name"]) for identity in identities]
    if names != sorted(names):
        raise ValueError(f"{label} must be sorted by name")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate artifact names")
    return identities


def _raw_artifact_identity_array(
    raw: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    if len(raw) > _MAX_SUITE_ARTIFACTS:
        raise ValueError(f"{label} exceeds hard artifact cap {_MAX_SUITE_ARTIFACTS}")
    identities: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        item_label = f"{label}[{index}]"
        identity = _identity(
            value,
            label=item_label,
            expected_keys=_RAW_ARTIFACT_IDENTITY_KEYS,
            named=True,
        )
        if not isinstance(value, Mapping):
            raise RuntimeError("validated raw artifact identity changed type")
        identity["role"] = _nonempty_string(
            value.get("role"),
            label=f"{item_label}.role",
            max_bytes=_MAX_SUITE_NAME_BYTES,
        )
        identities.append(identity)
    roles = [str(identity["role"]) for identity in identities]
    if roles != sorted(roles):
        raise ValueError(f"{label} must be sorted by role")
    if len(roles) != len(set(roles)):
        raise ValueError(f"{label} contains duplicate artifact roles")
    names = [str(identity["name"]) for identity in identities]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate artifact names")
    return identities


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    label: str,
) -> str:
    observed = _lowercase_sha256(
        value.get("manifest_self_sha256"),
        label=f"{label}.manifest_self_sha256",
    )
    without_hash = dict(value)
    without_hash.pop("manifest_self_sha256")
    if _sha256(_canonical_bytes(without_hash)) != observed:
        raise ValueError(f"{label} manifest_self_sha256 mismatch")
    return observed


def _validate_suite_provenance(
    value: Mapping[str, Any],
    *,
    raw: bytes,
    path: Path,
) -> tuple[str, str, Mapping[str, Any], int]:
    label = f"suite provenance {path}"
    _exact_keys(value, _SUITE_PROVENANCE_KEYS, label=label)
    if (
        value.get("kind") != SUITE_PROVENANCE_KIND
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != SUITE_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{label} must be {SUITE_PROVENANCE_KIND!r} "
            f"schema_version {SUITE_PROVENANCE_SCHEMA_VERSION}"
        )
    if value.get("status") != _SUITE_STATUS:
        raise ValueError(f"{label}.status is not a frozen prompt-only suite")
    manifest_self_sha256 = _validate_self_hash(value, label=label)
    if raw != _canonical_bytes(value):
        raise ValueError(f"{label} must use canonical JSON bytes")

    suite = value.get("suite")
    if not isinstance(suite, Mapping):
        raise ValueError(f"{label}.suite must be an object")
    _exact_keys(suite, _SUITE_KEYS, label=f"{label}.suite")
    name = _nonempty_string(
        suite.get("name"),
        label=f"{label}.suite.name",
        max_bytes=_MAX_SUITE_NAME_BYTES,
    )
    if _SUITE_NAME.fullmatch(name) is None:
        raise ValueError(f"{label}.suite.name contains unsupported characters")
    for key in ("benchmark", "revision", "split"):
        _nonempty_string(suite.get(key), label=f"{label}.suite.{key}")
    adapter = suite.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError(f"{label}.suite.adapter must be an object")
    _exact_keys(adapter, _ADAPTER_KEYS, label=f"{label}.suite.adapter")
    _nonempty_string(
        adapter.get("name"),
        label=f"{label}.suite.adapter.name",
    )
    adapter_version = _nonempty_string(
        adapter.get("version"),
        label=f"{label}.suite.adapter.version",
    )

    contract = value.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{label}.contract must be an object")
    _exact_keys(contract, _CONTRACT_IDENTITY_KEYS, label=f"{label}.contract")
    _relative_path(contract.get("path"), label=f"{label}.contract.path")
    _identity(
        {"bytes": contract.get("bytes"), "sha256": contract.get("sha256")},
        label=f"{label}.contract",
        expected_keys=frozenset({"bytes", "sha256"}),
    )

    benchmark_plan = value.get("benchmark_plan")
    if not isinstance(benchmark_plan, Mapping):
        raise ValueError(f"{label}.benchmark_plan must be an object")
    _exact_keys(
        benchmark_plan,
        _BENCHMARK_PLAN_IDENTITY_KEYS,
        label=f"{label}.benchmark_plan",
    )
    _identity(
        {
            "name": benchmark_plan.get("name"),
            "bytes": benchmark_plan.get("bytes"),
            "sha256": benchmark_plan.get("sha256"),
        },
        label=f"{label}.benchmark_plan",
        expected_keys=_ARTIFACT_IDENTITY_KEYS,
        named=True,
    )
    if benchmark_plan.get("plan_kind") != "localagent_external_benchmark_plan":
        raise ValueError(f"{label}.benchmark_plan.plan_kind is unsupported")
    if benchmark_plan.get("plan_schema_version") != 1 or isinstance(
        benchmark_plan.get("plan_schema_version"),
        bool,
    ):
        raise ValueError(f"{label}.benchmark_plan.plan_schema_version is invalid")
    _lowercase_sha256(
        benchmark_plan.get("suite_entry_sha256"),
        label=f"{label}.benchmark_plan.suite_entry_sha256",
    )

    sources = _named_identity_array(
        value.get("sources"),
        label=f"{label}.sources",
        expected_keys=_SOURCE_IDENTITY_KEYS,
        records=True,
    )
    raw_artifacts = _raw_artifact_identity_array(
        value.get("raw_artifacts"),
        label=f"{label}.raw_artifacts",
    )
    adapter_provenance_raw = value.get("adapter_provenance")
    if not isinstance(adapter_provenance_raw, list) or not adapter_provenance_raw:
        raise ValueError(f"{label}.adapter_provenance must be a non-empty array")
    if len(adapter_provenance_raw) > _MAX_SUITE_ARTIFACTS:
        raise ValueError(
            f"{label}.adapter_provenance exceeds hard artifact cap "
            f"{_MAX_SUITE_ARTIFACTS}"
        )
    adapter_provenance: list[dict[str, Any]] = []
    source_identities = {
        (
            int(source["bytes"]),
            str(source["sha256"]),
            int(source["records"]),
        )
        for source in sources
    }
    for index, raw_identity in enumerate(adapter_provenance_raw):
        item_label = f"{label}.adapter_provenance[{index}]"
        if not isinstance(raw_identity, Mapping):
            raise ValueError(f"{item_label} must be an object")
        _exact_keys(
            raw_identity,
            _ADAPTER_PROVENANCE_IDENTITY_KEYS,
            label=item_label,
        )
        identity = _identity(
            {
                "name": raw_identity.get("name"),
                "bytes": raw_identity.get("bytes"),
                "sha256": raw_identity.get("sha256"),
            },
            label=item_label,
            expected_keys=_ARTIFACT_IDENTITY_KEYS,
            named=True,
        )
        declared_adapter = _nonempty_string(
            raw_identity.get("adapter"),
            label=f"{item_label}.adapter",
        )
        if declared_adapter != adapter_version:
            raise ValueError(f"{item_label}.adapter disagrees with suite.adapter")
        _nonempty_string(
            raw_identity.get("audit_kind"),
            label=f"{item_label}.audit_kind",
        )
        _positive_int(
            raw_identity.get("audit_schema_version"),
            label=f"{item_label}.audit_schema_version",
        )
        bound = raw_identity.get("bound_prompt_source")
        if not isinstance(bound, Mapping):
            raise ValueError(f"{item_label}.bound_prompt_source must be an object")
        _exact_keys(
            bound,
            _BOUND_PROMPT_SOURCE_KEYS,
            label=f"{item_label}.bound_prompt_source",
        )
        bound_bytes = _positive_int(
            bound.get("bytes"),
            label=f"{item_label}.bound_prompt_source.bytes",
        )
        bound_sha256 = _lowercase_sha256(
            bound.get("sha256"),
            label=f"{item_label}.bound_prompt_source.sha256",
        )
        bound_records = _positive_int(
            bound.get("records"),
            label=f"{item_label}.bound_prompt_source.records",
        )
        if (bound_bytes, bound_sha256, bound_records) not in source_identities:
            raise ValueError(
                f"{item_label}.bound_prompt_source is not a declared source"
            )
        adapter_provenance.append(identity)
    adapter_names = [str(identity["name"]) for identity in adapter_provenance]
    if adapter_names != sorted(adapter_names):
        raise ValueError(f"{label}.adapter_provenance must be sorted by name")
    if len(adapter_names) != len(set(adapter_names)):
        raise ValueError(f"{label}.adapter_provenance contains duplicate names")

    evidence = _named_identity_array(
        value.get("license_evidence"),
        label=f"{label}.license_evidence",
        expected_keys=_ARTIFACT_IDENTITY_KEYS,
    )
    all_artifact_names = [
        *(str(identity["name"]) for identity in sources),
        *adapter_names,
        *(str(identity["name"]) for identity in evidence),
        *(str(identity["name"]) for identity in raw_artifacts),
    ]
    if len(all_artifact_names) != len(set(all_artifact_names)):
        raise ValueError(f"{label} contains duplicate artifact names")

    limits = value.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError(f"{label}.limits must be an object")
    _exact_keys(limits, _LIMIT_KEYS, label=f"{label}.limits")
    normalized_limits = {
        key: _positive_int(limits.get(key), label=f"{label}.limits.{key}")
        for key in _LIMIT_KEYS
    }
    if normalized_limits["hard_max_output_bytes"] > _MAX_SUITE_OUTPUT_BYTES:
        raise ValueError(f"{label}.limits.hard_max_output_bytes exceeds composer cap")
    if normalized_limits["max_rows"] > _MAX_SUITE_OUTPUT_RECORDS:
        raise ValueError(f"{label}.limits.max_rows exceeds composer cap")
    for count, hard_key, item_label in (
        (len(sources), "hard_max_source_artifacts", "sources"),
        (
            len(adapter_provenance),
            "hard_max_adapter_provenance_artifacts",
            "adapter_provenance",
        ),
        (
            len(evidence),
            "hard_max_license_evidence_artifacts",
            "license_evidence",
        ),
    ):
        if count > normalized_limits[hard_key]:
            raise ValueError(f"{label}.{item_label} exceeds its declared hard cap")

    deduplication = value.get("deduplication_audit")
    if not isinstance(deduplication, Mapping):
        raise ValueError(f"{label}.deduplication_audit must be an object")
    _exact_keys(
        deduplication,
        _DEDUPLICATION_KEYS,
        label=f"{label}.deduplication_audit",
    )
    if deduplication.get("method") != _DEDUPLICATION_METHOD:
        raise ValueError(f"{label}.deduplication_audit.method is unsupported")
    if deduplication.get("representative") != _DEDUPLICATION_REPRESENTATIVE:
        raise ValueError(
            f"{label}.deduplication_audit.representative is unsupported"
        )
    input_rows = _positive_int(
        deduplication.get("input_rows"),
        label=f"{label}.deduplication_audit.input_rows",
    )
    unique_rows = _positive_int(
        deduplication.get("unique_normalized_prompts"),
        label=f"{label}.deduplication_audit.unique_normalized_prompts",
    )
    removed_rows = _nonnegative_int(
        deduplication.get("normalized_prompt_duplicates_removed"),
        label=f"{label}.deduplication_audit.normalized_prompt_duplicates_removed",
    )
    if input_rows != sum(int(source["records"]) for source in sources):
        raise ValueError(f"{label}.deduplication_audit input row count mismatch")
    if unique_rows + removed_rows != input_rows:
        raise ValueError(f"{label}.deduplication_audit row accounting mismatch")
    for key in (
        "normalized_prompt_set_sha256",
        "input_source_case_id_hashes_sha256",
    ):
        _lowercase_sha256(
            deduplication.get(key),
            label=f"{label}.deduplication_audit.{key}",
        )

    output = value.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"{label}.output must be an object")
    _exact_keys(output, _OUTPUT_KEYS, label=f"{label}.output")
    _relative_path(output.get("path"), label=f"{label}.output.path")
    if output.get("format") != "canonical_jsonl":
        raise ValueError(f"{label}.output.format must be 'canonical_jsonl'")
    output_bytes = _positive_int(
        output.get("bytes"),
        label=f"{label}.output.bytes",
    )
    _lowercase_sha256(output.get("sha256"), label=f"{label}.output.sha256")
    output_records = _positive_int(
        output.get("records"),
        label=f"{label}.output.records",
    )
    if output.get("fields") != ["prompt", "source_case_id_sha256"]:
        raise ValueError(f"{label}.output.fields is not prompt-only")
    if output_records != unique_rows:
        raise ValueError(f"{label}.output.records disagrees with deduplication audit")
    if output_records > min(
        normalized_limits["max_rows"],
        _MAX_SUITE_OUTPUT_RECORDS,
    ):
        raise ValueError(f"{label}.output.records exceeds its declared cap")
    if output_bytes > min(
        normalized_limits["hard_max_output_bytes"],
        _MAX_SUITE_OUTPUT_BYTES,
    ):
        raise ValueError(f"{label}.output.bytes exceeds its declared hard cap")

    isolation = value.get("isolation")
    if not isinstance(isolation, Mapping):
        raise ValueError(f"{label}.isolation must be an object")
    _exact_keys(isolation, _ISOLATION_KEYS, label=f"{label}.isolation")
    expected_isolation = {
        "purpose": _ISOLATION_PURPOSE,
        "prompt_only": True,
        "contains_labels_or_expected_outputs": False,
        "fresh_labeled_evaluation_evidence": False,
        "benchmark_score_evidence": False,
        "permitted_training_use": _PERMITTED_TRAINING_USE,
    }
    for key, expected in expected_isolation.items():
        if isolation.get(key) != expected or (
            isinstance(expected, bool) and isolation.get(key) is not expected
        ):
            raise ValueError(f"{label}.isolation.{key} violates prompt-only isolation")
    _nonempty_string(
        isolation.get("limitations"),
        label=f"{label}.isolation.limitations",
    )
    return name, manifest_self_sha256, output, normalized_limits["hard_max_prompt_bytes"]


def _verify_bound_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    max_bytes: int,
    label: str,
) -> None:
    if expected_bytes > max_bytes:
        raise ValueError(f"{label} declared bytes exceed hard cap {max_bytes}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or is not a regular non-symlink file")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte-size disagrees with suite provenance")
    observed_bytes = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise ValueError(f"{label} exceeds hard cap {max_bytes}")
            digest.update(chunk)
    if observed_bytes != expected_bytes:
        raise ValueError(f"{label} byte-size disagrees with suite provenance")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 disagrees with suite provenance")


def _verify_output(
    path: Path,
    *,
    declared: Mapping[str, Any],
    max_prompt_bytes: int,
    label: str,
) -> tuple[int, str, int]:
    if not path.is_file():
        raise ValueError(f"{label} is missing or is not a file: {path}")
    expected_bytes = int(declared["bytes"])
    expected_sha256 = str(declared["sha256"])
    expected_records = int(declared["records"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte-size disagrees with suite provenance")

    observed_bytes = 0
    observed_records = 0
    observed_source_ids: set[str] = set()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            raw = handle.readline(_MAX_OUTPUT_RECORD_BYTES + 1)
            if not raw:
                break
            observed_records += 1
            observed_bytes += len(raw)
            if len(raw) > _MAX_OUTPUT_RECORD_BYTES:
                raise ValueError(
                    f"{label}:{observed_records} exceeds hard record byte cap "
                    f"{_MAX_OUTPUT_RECORD_BYTES}"
                )
            if observed_bytes > min(expected_bytes, _MAX_SUITE_OUTPUT_BYTES):
                raise ValueError(f"{label} exceeds its declared byte identity")
            if observed_records > min(
                expected_records,
                _MAX_SUITE_OUTPUT_RECORDS,
            ):
                raise ValueError(f"{label} exceeds its declared record count")
            digest.update(raw)
            row_label = f"{label}:{observed_records}"
            row = _strict_json_loads(raw, label=row_label)
            if not isinstance(row, Mapping):
                raise ValueError(f"{row_label} must be an object")
            _exact_keys(row, _OUTPUT_ROW_KEYS, label=row_label)
            prompt = _nonempty_string(
                row.get("prompt"),
                label=f"{row_label}.prompt",
            )
            if len(prompt.encode("utf-8")) > max_prompt_bytes:
                raise ValueError(f"{row_label}.prompt exceeds suite prompt cap")
            source_id = _lowercase_sha256(
                row.get("source_case_id_sha256"),
                label=f"{row_label}.source_case_id_sha256",
            )
            if source_id in observed_source_ids:
                raise ValueError(f"{row_label} repeats source_case_id_sha256")
            observed_source_ids.add(source_id)
            if raw != _canonical_bytes(dict(row)):
                raise ValueError(f"{row_label} must use canonical JSONL bytes")

    observed_sha256 = digest.hexdigest()
    if observed_bytes != expected_bytes:
        raise ValueError(f"{label} byte-size disagrees with suite provenance")
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 disagrees with suite provenance")
    if observed_records != expected_records:
        raise ValueError(f"{label} record count disagrees with suite provenance")
    return observed_bytes, observed_sha256, observed_records


def _verified_suite_provenance(path: str | Path) -> _VerifiedSuite:
    supplied_path = Path(path)
    provenance_path = supplied_path.resolve()
    value, raw = _read_json_object(
        supplied_path,
        label="suite provenance",
        max_bytes=_MAX_SUITE_PROVENANCE_BYTES,
    )
    name, manifest_self_sha256, output, max_prompt_bytes = (
        _validate_suite_provenance(
            value,
            raw=raw,
            path=provenance_path,
        )
    )
    contract = value["contract"]
    if not isinstance(contract, Mapping):
        raise RuntimeError("validated suite provenance contract changed type")
    contract_path = (
        provenance_path.parent
        / _relative_path(
            contract.get("path"),
            label=f"suite provenance {provenance_path}.contract.path",
        )
    ).resolve()
    contract_bytes = _positive_int(
        contract.get("bytes"),
        label=f"suite provenance {provenance_path}.contract.bytes",
    )
    contract_sha256 = _lowercase_sha256(
        contract.get("sha256"),
        label=f"suite provenance {provenance_path}.contract.sha256",
    )
    _verify_bound_file(
        contract_path,
        expected_bytes=contract_bytes,
        expected_sha256=contract_sha256,
        max_bytes=_MAX_SUITE_CONTRACT_BYTES,
        label=f"suite {name!r} contract",
    )
    output_path = (
        provenance_path.parent
        / _relative_path(
            output.get("path"),
            label=f"suite provenance {provenance_path}.output.path",
        )
    ).resolve()
    output_bytes, output_sha256, output_records = _verify_output(
        output_path,
        declared=output,
        max_prompt_bytes=max_prompt_bytes,
        label=f"suite {name!r} output",
    )
    rebuilt = verify_evaluation_denylist_suite(supplied_path)
    final_value, final_raw = _read_json_object(
        supplied_path,
        label="suite provenance",
        max_bytes=_MAX_SUITE_PROVENANCE_BYTES,
    )
    if final_value != rebuilt or final_raw != raw:
        raise RuntimeError(
            "reverified suite provenance differs from its on-disk manifest"
        )
    _verify_bound_file(
        contract_path,
        expected_bytes=contract_bytes,
        expected_sha256=contract_sha256,
        max_bytes=_MAX_SUITE_CONTRACT_BYTES,
        label=f"suite {name!r} contract",
    )
    final_output = _verify_output(
        output_path,
        declared=output,
        max_prompt_bytes=max_prompt_bytes,
        label=f"suite {name!r} output",
    )
    if final_output != (output_bytes, output_sha256, output_records):
        raise RuntimeError(f"suite {name!r} output changed during verification")
    benchmark_plan = value["benchmark_plan"]
    if not isinstance(benchmark_plan, Mapping):
        raise RuntimeError("validated suite provenance benchmark_plan changed type")
    return _VerifiedSuite(
        name=name,
        provenance_path=provenance_path,
        provenance_bytes=raw,
        provenance_sha256=_sha256(raw),
        manifest_self_sha256=manifest_self_sha256,
        benchmark_plan_bytes=_positive_int(
            benchmark_plan.get("bytes"),
            label=f"suite provenance {provenance_path}.benchmark_plan.bytes",
        ),
        benchmark_plan_sha256=_lowercase_sha256(
            benchmark_plan.get("sha256"),
            label=f"suite provenance {provenance_path}.benchmark_plan.sha256",
        ),
        output_path=output_path,
        output_bytes=output_bytes,
        output_sha256=output_sha256,
        output_records=output_records,
    )


def _validate_suite_set(suites: Sequence[_VerifiedSuite], *, destination: Path) -> None:
    names = [suite.name for suite in suites]
    if len(names) != len(set(names)):
        raise ValueError("suite provenance inputs contain duplicate suite names")
    provenance_paths = [suite.provenance_path for suite in suites]
    if len(provenance_paths) != len(set(provenance_paths)):
        raise ValueError("the same suite provenance path was supplied more than once")
    output_paths = [suite.output_path for suite in suites]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("suite provenance inputs bind duplicate output paths")
    protected_paths = set(provenance_paths) | set(output_paths)
    if destination.resolve() in protected_paths:
        raise ValueError("list manifest output must be distinct from suite inputs")
    if set(provenance_paths) & set(output_paths):
        raise ValueError("suite provenance and prompt output paths must be distinct")
    benchmark_plans = {
        (suite.benchmark_plan_bytes, suite.benchmark_plan_sha256) for suite in suites
    }
    if len(benchmark_plans) != 1:
        raise ValueError(
            "suite provenance inputs must all bind the same benchmark plan identity"
        )

    total_provenance_bytes = sum(len(suite.provenance_bytes) for suite in suites)
    if total_provenance_bytes > _MAX_TOTAL_SUITE_PROVENANCE_BYTES:
        raise ValueError(
            "suite provenance inputs exceed hard total byte cap "
            f"{_MAX_TOTAL_SUITE_PROVENANCE_BYTES}"
        )
    total_output_bytes = sum(suite.output_bytes for suite in suites)
    if total_output_bytes > _MAX_TOTAL_OUTPUT_BYTES:
        raise ValueError(
            f"suite outputs exceed hard total byte cap {_MAX_TOTAL_OUTPUT_BYTES}"
        )
    total_output_records = sum(suite.output_records for suite in suites)
    if total_output_records > _MAX_TOTAL_OUTPUT_RECORDS:
        raise ValueError(
            "suite outputs exceed hard total record cap "
            f"{_MAX_TOTAL_OUTPUT_RECORDS}"
        )


def _matches_payload(path: Path, payload: bytes) -> bool:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != len(payload)
    ):
        return False
    with path.open("rb") as handle:
        return handle.read(len(payload) + 1) == payload


def _assert_existing_or_absent(path: Path, payload: bytes) -> None:
    if path.exists() and not _matches_payload(path, payload):
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


def build_evaluation_denylist_manifest(
    suite_provenance_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build one canonical list manifest from verified suite provenance files."""

    if isinstance(suite_provenance_paths, (str, Path)):
        raise TypeError("suite_provenance_paths must be a sequence of paths")
    paths = list(suite_provenance_paths)
    if not paths:
        raise ValueError("at least one suite provenance path is required")
    if len(paths) > _MAX_SUITES:
        raise ValueError(f"suite provenance inputs exceed hard cap {_MAX_SUITES}")
    resolved_paths = [Path(path).resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("the same suite provenance path was supplied more than once")

    destination = Path(output_path)
    verified_suites: list[_VerifiedSuite] = []
    total_provenance_bytes = 0
    total_output_bytes = 0
    total_output_records = 0
    for path in paths:
        suite = _verified_suite_provenance(path)
        verified_suites.append(suite)
        total_provenance_bytes += len(suite.provenance_bytes)
        total_output_bytes += suite.output_bytes
        total_output_records += suite.output_records
        if total_provenance_bytes > _MAX_TOTAL_SUITE_PROVENANCE_BYTES:
            raise ValueError(
                "suite provenance inputs exceed hard total byte cap "
                f"{_MAX_TOTAL_SUITE_PROVENANCE_BYTES}"
            )
        if total_output_bytes > _MAX_TOTAL_OUTPUT_BYTES:
            raise ValueError(
                "suite outputs exceed hard total byte cap "
                f"{_MAX_TOTAL_OUTPUT_BYTES}"
            )
        if total_output_records > _MAX_TOTAL_OUTPUT_RECORDS:
            raise ValueError(
                "suite outputs exceed hard total record cap "
                f"{_MAX_TOTAL_OUTPUT_RECORDS}"
            )
    suites = sorted(verified_suites, key=lambda suite: suite.name)
    _validate_suite_set(suites, destination=destination)
    suite_rows: list[dict[str, Any]] = []
    for suite in suites:
        output_relative_path = _relative_path(
            _portable_path(
                suite.output_path,
                relative_to=destination.parent,
            ),
            label=f"suite {suite.name!r} generated output path",
        )
        provenance_relative_path = _relative_path(
            _portable_path(
                suite.provenance_path,
                relative_to=destination.parent,
            ),
            label=f"suite {suite.name!r} generated provenance path",
        )
        suite_rows.append(
            {
                "name": suite.name,
                "path": output_relative_path,
                "bytes": suite.output_bytes,
                "sha256": suite.output_sha256,
                "provenance": {
                    "path": provenance_relative_path,
                    "bytes": len(suite.provenance_bytes),
                    "sha256": suite.provenance_sha256,
                    "manifest_self_sha256": suite.manifest_self_sha256,
                },
            }
        )
    manifest_without_hash = {
        "kind": MANIFEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "required_suites": [suite.name for suite in suites],
        "suites": suite_rows,
    }
    manifest_self_sha256 = _sha256(_canonical_bytes(manifest_without_hash))
    manifest = {
        **manifest_without_hash,
        "manifest_self_sha256": manifest_self_sha256,
    }
    payload = _canonical_bytes(manifest)
    if len(payload) > _MAX_LIST_MANIFEST_BYTES:
        raise ValueError(
            f"canonical list manifest exceeds hard byte cap {_MAX_LIST_MANIFEST_BYTES}"
        )

    _assert_existing_or_absent(destination, payload)
    reverified = [
        _verified_suite_provenance(suite.provenance_path) for suite in suites
    ]
    if reverified != suites:
        raise ValueError("suite provenance inputs changed while composing the list manifest")
    _publish_atomic(destination, payload)
    return json.loads(payload)


def verify_evaluation_denylist_manifest(
    path: str | Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Strictly verify a composed list manifest and every bound suite artifact."""

    manifest_path = Path(path)
    manifest, raw = _read_json_object(
        manifest_path,
        label="evaluation denylist list manifest",
        max_bytes=_MAX_LIST_MANIFEST_BYTES,
    )
    label = f"evaluation denylist list manifest {manifest_path}"
    _exact_keys(manifest, _LIST_KEYS, label=label)
    if (
        manifest.get("kind") != MANIFEST_KIND
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(
            f"{label} must be {MANIFEST_KIND!r} schema_version {SCHEMA_VERSION}"
        )
    list_self_sha256 = _validate_self_hash(manifest, label=label)
    if raw != _canonical_bytes(manifest):
        raise ValueError(f"{label} must use canonical JSON bytes")

    raw_suites = manifest.get("suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ValueError(f"{label}.suites must be a non-empty array")
    if len(raw_suites) > _MAX_SUITES:
        raise ValueError(f"{label}.suites exceeds hard cap {_MAX_SUITES}")

    declared_names: list[str] = []
    declared_output_bytes = 0
    declared_provenance_bytes = 0
    for index, raw_suite in enumerate(raw_suites):
        suite_label = f"{label}.suites[{index}]"
        if not isinstance(raw_suite, Mapping):
            raise ValueError(f"{suite_label} must be an object")
        _exact_keys(raw_suite, _LIST_SUITE_KEYS, label=suite_label)
        name = _nonempty_string(
            raw_suite.get("name"),
            label=f"{suite_label}.name",
            max_bytes=_MAX_SUITE_NAME_BYTES,
        )
        if _SUITE_NAME.fullmatch(name) is None:
            raise ValueError(f"{suite_label}.name contains unsupported characters")
        declared_names.append(name)
        _relative_path(raw_suite.get("path"), label=f"{suite_label}.path")
        declared_output_bytes += _positive_int(
            raw_suite.get("bytes"),
            label=f"{suite_label}.bytes",
        )
        if declared_output_bytes > _MAX_TOTAL_OUTPUT_BYTES:
            raise ValueError(
                f"{label}.suites exceed hard total output byte cap "
                f"{_MAX_TOTAL_OUTPUT_BYTES}"
            )
        _lowercase_sha256(
            raw_suite.get("sha256"),
            label=f"{suite_label}.sha256",
        )
        raw_provenance = raw_suite.get("provenance")
        if not isinstance(raw_provenance, Mapping):
            raise ValueError(f"{suite_label}.provenance must be an object")
        _exact_keys(
            raw_provenance,
            _LIST_PROVENANCE_KEYS,
            label=f"{suite_label}.provenance",
        )
        _relative_path(
            raw_provenance.get("path"),
            label=f"{suite_label}.provenance.path",
        )
        provenance_bytes = _positive_int(
            raw_provenance.get("bytes"),
            label=f"{suite_label}.provenance.bytes",
        )
        if provenance_bytes > _MAX_SUITE_PROVENANCE_BYTES:
            raise ValueError(
                f"{suite_label}.provenance.bytes exceeds hard cap "
                f"{_MAX_SUITE_PROVENANCE_BYTES}"
            )
        declared_provenance_bytes += provenance_bytes
        if declared_provenance_bytes > _MAX_TOTAL_SUITE_PROVENANCE_BYTES:
            raise ValueError(
                f"{label}.suites exceed hard total provenance byte cap "
                f"{_MAX_TOTAL_SUITE_PROVENANCE_BYTES}"
            )
        _lowercase_sha256(
            raw_provenance.get("sha256"),
            label=f"{suite_label}.provenance.sha256",
        )
        _lowercase_sha256(
            raw_provenance.get("manifest_self_sha256"),
            label=f"{suite_label}.provenance.manifest_self_sha256",
        )
    if declared_names != sorted(declared_names):
        raise ValueError(f"{label}.suites must be sorted by name")
    if len(declared_names) != len(set(declared_names)):
        raise ValueError(f"{label}.suites contains duplicate suite names")

    raw_required = manifest.get("required_suites")
    if not isinstance(raw_required, list):
        raise ValueError(f"{label}.required_suites must be an array")
    required = [
        _nonempty_string(
            name,
            label=f"{label}.required_suites[{index}]",
            max_bytes=_MAX_SUITE_NAME_BYTES,
        )
        for index, name in enumerate(raw_required)
    ]
    if required != declared_names:
        raise ValueError(
            f"{label}.required_suites must exactly equal all sorted suite names"
        )

    artifacts: list[dict[str, object]] = []
    verified_suites: list[_VerifiedSuite] = []
    observed_names: set[str] = set()
    observed_provenance_paths: set[Path] = set()
    observed_output_paths: set[Path] = set()
    verified_output_records = 0
    for index, raw_suite in enumerate(raw_suites):
        suite_label = f"{label}.suites[{index}]"
        if not isinstance(raw_suite, Mapping):
            raise ValueError(f"{suite_label} must be an object")
        _exact_keys(raw_suite, _LIST_SUITE_KEYS, label=suite_label)
        name = _nonempty_string(
            raw_suite.get("name"),
            label=f"{suite_label}.name",
            max_bytes=_MAX_SUITE_NAME_BYTES,
        )
        if _SUITE_NAME.fullmatch(name) is None:
            raise ValueError(f"{suite_label}.name contains unsupported characters")
        if name in observed_names:
            raise ValueError(f"{label} contains duplicate suite name {name!r}")
        observed_names.add(name)
        declared_path = _relative_path(
            raw_suite.get("path"),
            label=f"{suite_label}.path",
        )
        expected_bytes = _positive_int(
            raw_suite.get("bytes"),
            label=f"{suite_label}.bytes",
        )
        expected_sha256 = _lowercase_sha256(
            raw_suite.get("sha256"),
            label=f"{suite_label}.sha256",
        )

        raw_provenance = raw_suite.get("provenance")
        if not isinstance(raw_provenance, Mapping):
            raise ValueError(f"{suite_label}.provenance must be an object")
        _exact_keys(
            raw_provenance,
            _LIST_PROVENANCE_KEYS,
            label=f"{suite_label}.provenance",
        )
        declared_provenance_path = _relative_path(
            raw_provenance.get("path"),
            label=f"{suite_label}.provenance.path",
        )
        expected_provenance_bytes = _positive_int(
            raw_provenance.get("bytes"),
            label=f"{suite_label}.provenance.bytes",
        )
        if expected_provenance_bytes > _MAX_SUITE_PROVENANCE_BYTES:
            raise ValueError(
                f"{suite_label}.provenance.bytes exceeds hard cap "
                f"{_MAX_SUITE_PROVENANCE_BYTES}"
            )
        expected_provenance_sha256 = _lowercase_sha256(
            raw_provenance.get("sha256"),
            label=f"{suite_label}.provenance.sha256",
        )
        expected_provenance_self_sha256 = _lowercase_sha256(
            raw_provenance.get("manifest_self_sha256"),
            label=f"{suite_label}.provenance.manifest_self_sha256",
        )

        declared_provenance_file = (
            manifest_path.parent / declared_provenance_path
        )
        declared_output_file = manifest_path.parent / declared_path
        provenance_path = declared_provenance_file.resolve()
        output_path = declared_output_file.resolve()
        if declared_provenance_path != _portable_path(
            provenance_path,
            relative_to=manifest_path.parent,
        ):
            raise ValueError(
                f"{suite_label}.provenance.path is not canonical and portable"
            )
        if declared_path != _portable_path(
            output_path,
            relative_to=manifest_path.parent,
        ):
            raise ValueError(f"{suite_label}.path is not canonical and portable")
        if provenance_path in observed_provenance_paths:
            raise ValueError(f"{label} contains duplicate suite provenance paths")
        if output_path in observed_output_paths:
            raise ValueError(f"{label} contains duplicate suite output paths")
        observed_provenance_paths.add(provenance_path)
        observed_output_paths.add(output_path)
        try:
            verified = _verified_suite_provenance(declared_provenance_file)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f"{label}: suite {name!r} provenance verification failed: {error}"
            ) from error
        if verified.name != name:
            raise ValueError(
                f"{suite_label}.name disagrees with suite provenance"
            )
        if len(verified.provenance_bytes) != expected_provenance_bytes:
            raise ValueError(
                f"{suite_label}.provenance byte-size mismatch"
            )
        if verified.provenance_sha256 != expected_provenance_sha256:
            raise ValueError(f"{suite_label}.provenance SHA-256 mismatch")
        if (
            verified.manifest_self_sha256
            != expected_provenance_self_sha256
        ):
            raise ValueError(
                f"{suite_label}.provenance manifest_self_sha256 mismatch"
            )
        if verified.output_path != output_path:
            raise ValueError(
                f"{suite_label}.path disagrees with suite provenance output binding"
            )
        if verified.output_bytes != expected_bytes:
            raise ValueError(f"{suite_label} output byte-size mismatch")
        if verified.output_sha256 != expected_sha256:
            raise ValueError(f"{suite_label} output SHA-256 mismatch")
        verified_output_records += verified.output_records
        if verified_output_records > _MAX_TOTAL_OUTPUT_RECORDS:
            raise ValueError(
                f"{label}.suites exceed hard total output record cap "
                f"{_MAX_TOTAL_OUTPUT_RECORDS}"
            )
        verified_suites.append(verified)
        artifacts.append(
            {
                "name": name,
                "path": str(manifest_path.parent / declared_path),
                "bytes": expected_bytes,
                "sha256": expected_sha256,
                "source": "list_manifest",
                "declared_path": declared_path,
                "provenance": {
                    "path": str(
                        manifest_path.parent / declared_provenance_path
                    ),
                    "declared_path": declared_provenance_path,
                    "bytes": expected_provenance_bytes,
                    "sha256": expected_provenance_sha256,
                    "manifest_self_sha256": (
                        expected_provenance_self_sha256
                    ),
                },
            }
        )

    if manifest_path.resolve() in observed_provenance_paths | observed_output_paths:
        raise ValueError(f"{label} must be distinct from every suite artifact")
    if observed_provenance_paths & observed_output_paths:
        raise ValueError(
            f"{label} suite provenance and output paths must be distinct"
        )
    _validate_suite_set(verified_suites, destination=manifest_path)

    _, final_raw = _read_json_object(
        manifest_path,
        label="evaluation denylist list manifest",
        max_bytes=_MAX_LIST_MANIFEST_BYTES,
    )
    if final_raw != raw:
        raise ValueError(f"{label} changed while it was being verified")
    for index, (before, raw_suite) in enumerate(
        zip(verified_suites, raw_suites, strict=True)
    ):
        raw_provenance = raw_suite["provenance"]
        if not isinstance(raw_provenance, Mapping):
            raise RuntimeError(f"{label}.suites[{index}].provenance changed type")
        declared_provenance_path = _relative_path(
            raw_provenance.get("path"),
            label=f"{label}.suites[{index}].provenance.path",
        )
        try:
            after = _verified_suite_provenance(
                manifest_path.parent / declared_provenance_path
            )
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f"{label}: suite {before.name!r} changed during verification: {error}"
            ) from error
        if after != before:
            raise ValueError(
                f"{label}: suite {before.name!r} changed during verification"
            )

    return artifacts, {
        "path": str(manifest_path),
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "kind": MANIFEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "manifest_self_sha256": list_self_sha256,
        "required_suites": required,
    }


__all__ = [
    "MANIFEST_KIND",
    "SCHEMA_VERSION",
    "build_evaluation_denylist_manifest",
    "verify_evaluation_denylist_manifest",
]
