"""Freeze, verify, and compare a fresh external action-evaluation slice.

No command downloads benchmark data.  See ``docs/paper/FRESH_EXTERNAL_EVAL_CONTRACT.md`` for the
required source-export and contract schemas.  Historical unversioned inputs remain a v1
normalized-call comparison; v2 paper evidence requires versioned raw whole outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    schema_matches,
    validate_tool_catalog,
)
from localagent.data.schema import ToolCall, ToolSpec
from localagent.eval.external_action_contract import (
    HARDENED_SCHEMA_VERSION,
    MANIFEST_KIND,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    freeze_external_action_slice,
    validate_frozen_slice,
    verify_declared_file_identity,
)
from localagent.eval.realtime import paired_clustered_exact_action_delta_ci
from localagent.eval.tool_eval import match_calls, parse_tool_output

_MAX_JSON_BYTES = 512 * 1024 * 1024
_MAX_MODEL_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_MAX_RAW_OUTPUT_CHARS = 1024 * 1024
_MAX_BOOTSTRAP_RESAMPLES = 1_000_000
_SHA256_HEX = frozenset("0123456789abcdef")

RAW_OUTPUT_RESULT_KIND = "localagent_external_raw_model_output_result"
RAW_OUTPUT_RESULT_SCHEMA_VERSION = 1
NORMALIZED_CALL_RESULT_KIND = "localagent_external_normalized_call_result"
NORMALIZED_CALL_RESULT_SCHEMA_VERSION = 1
RAW_OUTPUT_COMPARISON_KIND = "localagent_fresh_external_raw_output_action_comparison"
NORMALIZED_CALL_COMPARISON_KIND = "localagent_fresh_external_normalized_call_comparison"
LEGACY_COMPARISON_KIND = "localagent_fresh_external_action_comparison"
_RAW_OUTPUT_SEMANTICS = "raw_whole_output_strict_parse_tool_output_v1"
_NORMALIZED_OUTPUT_SEMANTICS = "adapter_supplied_normalized_calls_v1"
_LEGACY_OUTPUT_SEMANTICS = "legacy_unversioned_normalized_calls_v0"
_SUCCESSFUL_FINISH_REASONS = frozenset({"eos", "stop"})
_FINISH_REASONS = _SUCCESSFUL_FINISH_REASONS | frozenset(
    {"length", "max_new_tokens", "runtime_error", "cancelled"}
)

_LEGACY_RESULT_KEYS = frozenset({"system", "records"})
_RAW_RESULT_KEYS = frozenset({"kind", "schema_version", "system", "records"})
_LEGACY_SYSTEM_KEYS = frozenset({"name", "checkpoint_sha256", "bundle_sha256"})
_RAW_SYSTEM_KEYS = frozenset({"name", "checkpoint", "bundle"})
_ARTIFACT_IDENTITY_KEYS = frozenset({"path", "bytes", "sha256"})
_LEGACY_RECORD_REQUIRED_KEYS = frozenset(
    {"case_id", "task_cluster_id", "repetition", "predicted_calls"}
)
_RAW_RECORD_REQUIRED_KEYS = frozenset(
    {"case_id", "task_cluster_id", "repetition", "raw_output", "finish_reason"}
)
_RECORD_OPTIONAL_KEYS = frozenset({"success"})
_CALL_KEYS = frozenset({"name", "arguments"})
_MANIFEST_KEYS_V1 = frozenset(
    {
        "kind",
        "schema_version",
        "status",
        "contract",
        "source",
        "selection",
        "training_artifacts",
        "decontamination_audit",
        "analysis",
        "outputs",
        "isolation",
        "manifest_self_sha256",
    }
)
_MANIFEST_KEYS_V2 = _MANIFEST_KEYS_V1 | {"lineage_artifacts"}
_MANIFEST_ANALYSIS_KEYS = frozenset(
    {
        "bootstrap_resamples",
        "bootstrap_seed",
        "exact_action_noninferiority_margin",
        "estimand",
    }
)
_MANIFEST_OUTPUT_KEYS = frozenset({"slice", "prompt_only_denylist"})
_MANIFEST_OUTPUT_IDENTITY_KEYS = frozenset({"path", "bytes", "sha256", "cases"})
_MANIFEST_LINEAGE_REQUIRED_KEYS = frozenset(
    {
        "path",
        "bytes",
        "sha256",
        "stage",
        "name",
        "checkpoint_sha256",
        "lineage_sha256",
        "training_artifact_sha256",
        "conversation_prompt_contract",
    }
)
_MANIFEST_LINEAGE_OPTIONAL_KEYS = frozenset({"parent_checkpoint_sha256"})


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


def _strict_json(payload: bytes, *, label: str) -> Any:
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
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error


def _exact_mapping(value: Any, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _mapping_with_optional(
    value: Any,
    required: frozenset[str],
    optional: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


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


def _read_regular(path: Path, *, label: str, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
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
            raise ValueError(f"artifact exceeds {max_bytes} bytes: {path}")
        try:
            path_state = path.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} pathname changed while being read: {path}") from error
        if not _same_file_state(initial, path_state):
            raise RuntimeError(f"{label} changed while its descriptor was being bound: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"artifact exceeds {max_bytes} bytes: {path}")
        final = os.fstat(descriptor)
        final_path = path.lstat()
        if not _same_file_state(initial, final) or not _same_file_state(initial, final_path):
            raise RuntimeError(f"{label} changed while it was being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_object(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read_regular(path, label=label)
    payload = _strict_json(raw, label=label)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload, {
        "path": path.name,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _verified_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, identity = _read_object(path, label="manifest")
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("manifest schema_version is unsupported")
    payload = _exact_mapping(
        payload,
        _MANIFEST_KEYS_V2 if schema_version == HARDENED_SCHEMA_VERSION else _MANIFEST_KEYS_V1,
        label="manifest",
    )
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError("manifest kind/schema_version mismatch")
    observed_self_hash = payload.get("manifest_self_sha256")
    _require_sha256(observed_self_hash, label="manifest.manifest_self_sha256")
    without_hash = dict(payload)
    without_hash.pop("manifest_self_sha256")
    if _sha256(_canonical_bytes(without_hash)) != observed_self_hash:
        raise ValueError("manifest_self_sha256 mismatch")
    _validate_manifest_fields(payload)
    return payload, identity


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_manifest_fields(payload: dict[str, Any]) -> None:
    analysis = _exact_mapping(
        payload.get("analysis"),
        _MANIFEST_ANALYSIS_KEYS,
        label="manifest.analysis",
    )
    resamples = analysis["bootstrap_resamples"]
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or not 1 <= resamples <= _MAX_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            f"manifest.analysis.bootstrap_resamples must be in [1, {_MAX_BOOTSTRAP_RESAMPLES}]"
        )
    seed = analysis["bootstrap_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("manifest.analysis.bootstrap_seed must be an integer")
    margin = analysis["exact_action_noninferiority_margin"]
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not -1.0 <= float(margin) <= 1.0
    ):
        raise ValueError(
            "manifest.analysis.exact_action_noninferiority_margin must be finite and in [-1, 1]"
        )
    if not isinstance(analysis["estimand"], str) or not analysis["estimand"]:
        raise TypeError("manifest.analysis.estimand must be non-empty text")

    outputs = _exact_mapping(
        payload.get("outputs"),
        _MANIFEST_OUTPUT_KEYS,
        label="manifest.outputs",
    )
    for name, value in outputs.items():
        output = _exact_mapping(
            value,
            _MANIFEST_OUTPUT_IDENTITY_KEYS,
            label=f"manifest.outputs.{name}",
        )
        if not isinstance(output["path"], str) or not output["path"]:
            raise TypeError(f"manifest.outputs.{name}.path must be non-empty text")
        if (
            isinstance(output["bytes"], bool)
            or not isinstance(output["bytes"], int)
            or output["bytes"] < 0
        ):
            raise ValueError(f"manifest.outputs.{name}.bytes must be non-negative")
        _require_sha256(
            output["sha256"],
            label=f"manifest.outputs.{name}.sha256",
        )
        if (
            isinstance(output["cases"], bool)
            or not isinstance(output["cases"], int)
            or output["cases"] < 1
        ):
            raise ValueError(f"manifest.outputs.{name}.cases must be positive")

    if payload["schema_version"] != HARDENED_SCHEMA_VERSION:
        return
    lineage_artifacts = payload["lineage_artifacts"]
    if not isinstance(lineage_artifacts, list) or not lineage_artifacts:
        raise ValueError("manifest.lineage_artifacts must be a non-empty array")
    observed_stages: set[str] = set()
    observed_names: set[str] = set()
    validated_lineage: list[dict[str, Any]] = []
    for index, value in enumerate(lineage_artifacts):
        label = f"manifest.lineage_artifacts[{index}]"
        item = _mapping_with_optional(
            value,
            _MANIFEST_LINEAGE_REQUIRED_KEYS,
            _MANIFEST_LINEAGE_OPTIONAL_KEYS,
            label=label,
        )
        if item["stage"] not in {"pretrain", "midtrain", "sft", "rl"}:
            raise ValueError(f"{label}.stage is unsupported")
        observed_stages.add(item["stage"])
        if not isinstance(item["path"], str) or not item["path"]:
            raise TypeError(f"{label}.path must be non-empty text")
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
        ):
            raise ValueError(f"{label}.bytes must be non-negative")
        if not isinstance(item["name"], str) or not item["name"]:
            raise TypeError(f"{label}.name must be non-empty text")
        if item["name"] in observed_names:
            raise ValueError(f"duplicate manifest lineage name {item['name']!r}")
        observed_names.add(item["name"])
        for key in ("sha256", "checkpoint_sha256", "lineage_sha256"):
            _require_sha256(item[key], label=f"{label}.{key}")
        if "parent_checkpoint_sha256" in item:
            _require_sha256(
                item["parent_checkpoint_sha256"],
                label=f"{label}.parent_checkpoint_sha256",
            )
        hashes = item["training_artifact_sha256"]
        if not isinstance(hashes, list) or not hashes or len(set(hashes)) != len(hashes):
            raise ValueError(f"{label}.training_artifact_sha256 must contain unique hashes")
        for hash_index, digest in enumerate(hashes):
            _require_sha256(
                digest,
                label=f"{label}.training_artifact_sha256[{hash_index}]",
            )
        prompt_contract = item["conversation_prompt_contract"]
        if item["stage"] == "pretrain":
            if prompt_contract is not None:
                raise ValueError(f"{label}.conversation_prompt_contract must be null")
        elif prompt_contract != OPENAI_FULL_CATALOG_V1:
            raise ValueError(
                f"{label}.conversation_prompt_contract must be {OPENAI_FULL_CATALOG_V1!r}"
            )
        validated_lineage.append(item)
    if observed_stages != {"pretrain", "midtrain", "sft", "rl"}:
        raise ValueError("manifest.lineage_artifacts must cover all four stages")
    checkpoints_by_stage: dict[str, set[str]] = {}
    for stage in observed_stages:
        checkpoints_by_stage[stage] = {
            item["checkpoint_sha256"] for item in validated_lineage if item["stage"] == stage
        }
    parent_stage = {"midtrain": "pretrain", "sft": "midtrain", "rl": "sft"}
    for item in validated_lineage:
        stage = item["stage"]
        parent = item.get("parent_checkpoint_sha256")
        if stage == "pretrain":
            if parent is not None:
                raise ValueError("manifest pretrain lineage must not have a parent")
            continue
        if parent not in checkpoints_by_stage[parent_stage[stage]]:
            raise ValueError(
                f"manifest {stage} lineage parent is absent from {parent_stage[stage]}"
            )


def _validate_slice_identity(
    slice_path: Path,
    *,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, identity = _read_object(slice_path, label="frozen slice")
    declared = manifest.get("outputs", {}).get("slice")
    if not isinstance(declared, dict):
        raise TypeError("manifest outputs.slice is missing")
    if identity["bytes"] != declared.get("bytes") or identity["sha256"] != declared.get("sha256"):
        raise ValueError("frozen slice byte identity disagrees with manifest")
    validated = validate_frozen_slice(payload)
    if validated["schema_version"] != manifest["schema_version"]:
        raise ValueError("frozen slice schema_version disagrees with manifest")
    if len(validated["cases"]) != declared["cases"]:
        raise ValueError("frozen slice case count disagrees with manifest")
    return validated, identity


def _tool_calls(value: Any, *, label: str) -> list[ToolCall]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    calls: list[ToolCall] = []
    for index, call in enumerate(value):
        call = _exact_mapping(call, _CALL_KEYS, label=f"{label}[{index}]")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}[{index}].name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise TypeError(f"{label}[{index}].arguments must be an object")
        try:
            json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}[{index}].arguments must be finite JSON") from error
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def _legacy_system_metadata(value: Any, *, label: str) -> dict[str, Any]:
    value = _exact_mapping(value, _LEGACY_SYSTEM_KEYS, label=f"{label}.system")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.system.name must be a non-empty string")
    for key in ("checkpoint_sha256", "bundle_sha256"):
        _require_sha256(value.get(key), label=f"{label}.system.{key}")
    return value


def _raw_system_metadata(
    value: Any,
    *,
    result_path: Path,
    label: str,
) -> dict[str, Any]:
    value = _exact_mapping(value, _RAW_SYSTEM_KEYS, label=f"{label}.system")
    name = value["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.system.name must be a non-empty string")
    verified: dict[str, Any] = {"name": name}
    resolved: set[Path] = set()
    for key in ("checkpoint", "bundle"):
        declaration = _exact_mapping(
            value[key],
            _ARTIFACT_IDENTITY_KEYS,
            label=f"{label}.system.{key}",
        )
        path, identity = verify_declared_file_identity(
            declaration,
            base=result_path.parent,
            label=f"{label}.system.{key}",
            max_bytes=_MAX_MODEL_ARTIFACT_BYTES,
        )
        path_identity = path.resolve()
        if path_identity == result_path.resolve():
            raise ValueError(f"{label}.system.{key} must not reuse its result artifact")
        if path_identity in resolved:
            raise ValueError(f"{label}.system checkpoint and bundle must be distinct files")
        resolved.add(path_identity)
        verified[key] = identity
        verified[f"{key}_resolved_path"] = str(path_identity)
    return verified


def _frozen_tool_specs(case: dict[str, Any], *, label: str) -> tuple[ToolSpec, ...]:
    tools = tuple(
        ToolSpec(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
        )
        for tool in case["tools"]
    )
    validate_tool_catalog(tools, label=label)
    return tools


def _raw_prediction_score(
    raw_output: str,
    finish_reason: str,
    *,
    case: dict[str, Any],
    expected: list[ToolCall],
    label: str,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(raw_output, str):
        raise TypeError(f"{label}.raw_output must be text")
    if len(raw_output) > _MAX_RAW_OUTPUT_CHARS:
        raise ValueError(f"{label}.raw_output exceeds {_MAX_RAW_OUTPUT_CHARS} Unicode code points")
    if not isinstance(finish_reason, str) or finish_reason not in _FINISH_REASONS:
        raise ValueError(f"{label}.finish_reason must be one of {sorted(_FINISH_REASONS)}")
    parsed = parse_tool_output(raw_output)
    tools = _frozen_tool_specs(case, label=f"{label} frozen catalog")
    registry = {tool.name: tool for tool in tools}
    schema_valid = (
        parsed.format_valid
        and bool(parsed.calls)
        and all(
            call.name in registry and schema_matches(call.arguments, registry[call.name].parameters)
            for call in parsed.calls
        )
    )
    exact = (
        finish_reason in _SUCCESSFUL_FINISH_REASONS
        and schema_valid
        and match_calls(list(parsed.calls), expected)
    )
    return exact, {
        "finish_reason": finish_reason,
        "finish_reason_complete": finish_reason in _SUCCESSFUL_FINISH_REASONS,
        "format_valid": parsed.format_valid,
        "schema_valid": schema_valid,
        "tool_syntax_present": parsed.tool_syntax_present,
        "parse_errors": list(parsed.errors),
    }


def _scored_records(
    path: Path,
    *,
    frozen_cases: dict[str, dict[str, Any]],
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    payload, identity = _read_object(path, label=label)
    kind = payload.get("kind")
    if kind is None:
        payload = _exact_mapping(payload, _LEGACY_RESULT_KEYS, label=label)
        result_semantics = _LEGACY_OUTPUT_SEMANTICS
        system = _legacy_system_metadata(payload.get("system"), label=label)
    else:
        payload = _exact_mapping(payload, _RAW_RESULT_KEYS, label=label)
        if kind == RAW_OUTPUT_RESULT_KIND:
            expected_version = RAW_OUTPUT_RESULT_SCHEMA_VERSION
            result_semantics = _RAW_OUTPUT_SEMANTICS
            system = _raw_system_metadata(payload.get("system"), result_path=path, label=label)
        elif kind == NORMALIZED_CALL_RESULT_KIND:
            expected_version = NORMALIZED_CALL_RESULT_SCHEMA_VERSION
            result_semantics = _NORMALIZED_OUTPUT_SEMANTICS
            system = _legacy_system_metadata(payload.get("system"), label=label)
        else:
            raise ValueError(
                f"{label}.kind must be {RAW_OUTPUT_RESULT_KIND!r} or "
                f"{NORMALIZED_CALL_RESULT_KIND!r}"
            )
        if payload.get("schema_version") != expected_version:
            raise ValueError(f"{label} {kind!r} schema_version must be {expected_version}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{label}.records must be a non-empty array")

    scored: list[dict[str, Any]] = []
    observed_cases: set[str] = set()
    observed_opportunities: set[tuple[str, int]] = set()
    parse_outcomes = {
        "format_invalid": 0,
        "schema_invalid": 0,
        "incomplete_finish": 0,
    }
    for index, record in enumerate(raw_records):
        row_label = f"{label}.records[{index}]"
        record = _mapping_with_optional(
            record,
            (
                _RAW_RECORD_REQUIRED_KEYS
                if result_semantics == _RAW_OUTPUT_SEMANTICS
                else _LEGACY_RECORD_REQUIRED_KEYS
            ),
            _RECORD_OPTIONAL_KEYS,
            label=row_label,
        )
        case_id = record.get("case_id")
        cluster_id = record.get("task_cluster_id")
        if not isinstance(case_id, str) or case_id not in frozen_cases:
            raise ValueError(f"{row_label}.case_id is not in the frozen slice")
        expected_cluster = frozen_cases[case_id]["task_cluster_id"]
        if cluster_id != expected_cluster:
            raise ValueError(f"{row_label}.task_cluster_id disagrees with the frozen slice")
        repetition = record.get("repetition")
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0:
            raise ValueError(f"{row_label}.repetition must be a non-negative integer")
        opportunity = (case_id, repetition)
        if opportunity in observed_opportunities:
            raise ValueError(f"{row_label} duplicates case/repetition opportunity {opportunity!r}")
        observed_opportunities.add(opportunity)
        expected = _tool_calls(
            frozen_cases[case_id]["expected_calls"],
            label=f"frozen_cases[{case_id!r}].expected_calls",
        )
        row_audit: dict[str, Any] = {}
        if result_semantics == _RAW_OUTPUT_SEMANTICS:
            exact, row_audit = _raw_prediction_score(
                record["raw_output"],
                record["finish_reason"],
                case=frozen_cases[case_id],
                expected=expected,
                label=row_label,
            )
            if not row_audit["format_valid"]:
                parse_outcomes["format_invalid"] += 1
            if not row_audit["schema_valid"]:
                parse_outcomes["schema_invalid"] += 1
            if not row_audit["finish_reason_complete"]:
                parse_outcomes["incomplete_finish"] += 1
        else:
            predicted = _tool_calls(
                record.get("predicted_calls"), label=f"{row_label}.predicted_calls"
            )
            exact = match_calls(predicted, expected)
        reported = record.get("success")
        if reported is not None and (not isinstance(reported, bool) or reported is not exact):
            raise ValueError(f"{row_label}.success disagrees with independent AST scoring")
        scored.append(
            {
                "case_id": case_id,
                "task_cluster_id": cluster_id,
                "repetition": repetition,
                "success": exact,
                **row_audit,
            }
        )
        observed_cases.add(case_id)
    expected_cases = set(frozen_cases)
    if observed_cases != expected_cases:
        raise ValueError(
            f"{label} must cover every frozen case; missing={sorted(expected_cases - observed_cases)}"
        )
    public_system = dict(system)
    public_system.pop("checkpoint_resolved_path", None)
    public_system.pop("bundle_resolved_path", None)
    if result_semantics == _LEGACY_OUTPUT_SEMANTICS:
        # Preserve the exact historical comparison identity so honest v1 artifacts continue to
        # reproduce.  New collections must choose one of the explicit versioned result kinds.
        return (
            scored,
            {
                **identity,
                "system": public_system,
                "records": len(scored),
                "cases": len(observed_cases),
                "success_recomputed": True,
                "scoring": "order-insensitive exact tool name plus canonical JSON arguments",
            },
            result_semantics,
        )
    return (
        scored,
        {
            **identity,
            "result_contract": {
                "kind": (
                    RAW_OUTPUT_RESULT_KIND
                    if result_semantics == _RAW_OUTPUT_SEMANTICS
                    else NORMALIZED_CALL_RESULT_KIND
                ),
                "schema_version": (
                    RAW_OUTPUT_RESULT_SCHEMA_VERSION
                    if result_semantics == _RAW_OUTPUT_SEMANTICS
                    else NORMALIZED_CALL_RESULT_SCHEMA_VERSION
                ),
                "evaluation_semantics": result_semantics,
                "raw_model_output_observed": result_semantics == _RAW_OUTPUT_SEMANTICS,
            },
            "system": public_system,
            "records": len(scored),
            "cases": len(observed_cases),
            "success_recomputed": True,
            "scoring": (
                "strict whole-output parse_tool_output, frozen-catalog recursive schema validation, "
                "complete finish reason, then order-insensitive exact normalized calls"
                if result_semantics == _RAW_OUTPUT_SEMANTICS
                else (
                    "order-insensitive exact normalized calls supplied by an adapter; "
                    "not raw whole-output model evaluation"
                )
            ),
            **(
                {
                    "parse_outcomes": parse_outcomes,
                    "actual_artifact_identity_verified": True,
                }
                if result_semantics == _RAW_OUTPUT_SEMANTICS
                else {"actual_artifact_identity_verified": False}
            ),
        },
        result_semantics,
    )


def _publish(path: Path, payload: bytes) -> None:
    if path.exists():
        try:
            observed = _read_regular(
                path,
                label="existing comparison artifact",
                max_bytes=len(payload),
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"refusing to overwrite drifted comparison artifact: {path}"
            ) from error
        if observed != payload:
            raise RuntimeError(f"refusing to overwrite drifted comparison artifact: {path}")
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
            try:
                observed = _read_regular(
                    path,
                    label="concurrently created comparison artifact",
                    max_bytes=len(payload),
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    f"refusing to overwrite concurrently created comparison artifact: {path}"
                ) from error
            if observed != payload:
                raise RuntimeError(
                    f"refusing to overwrite concurrently created comparison artifact: {path}"
                )
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    try:
        observed = _read_regular(
            path,
            label="published comparison artifact",
            max_bytes=len(payload),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"published comparison artifact failed verification: {path}") from error
    if observed != payload:
        raise RuntimeError(f"published comparison artifact failed verification: {path}")


def _freeze(args: argparse.Namespace) -> None:
    manifest = freeze_external_action_slice(
        args.contract,
        slice_path=args.slice,
        denylist_path=args.denylist,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))


def _compare(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    slice_path = Path(args.slice)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    input_paths = {
        manifest_path.resolve(),
        slice_path.resolve(),
        baseline_path.resolve(),
        candidate_path.resolve(),
    }
    if len(input_paths) != 4:
        raise ValueError("manifest, slice, baseline, and candidate must be distinct files")
    manifest, manifest_identity = _verified_manifest(manifest_path)
    frozen, slice_identity = _validate_slice_identity(slice_path, manifest=manifest)
    frozen_cases = {case["case_id"]: case for case in frozen["cases"]}
    baseline, baseline_identity, baseline_semantics = _scored_records(
        baseline_path,
        frozen_cases=frozen_cases,
        label="baseline",
    )
    candidate, candidate_identity, candidate_semantics = _scored_records(
        candidate_path,
        frozen_cases=frozen_cases,
        label="candidate",
    )
    if baseline_semantics != candidate_semantics:
        raise ValueError(
            "baseline and candidate must use the same result contract; refusing to mix raw "
            "whole-output evaluation with adapter-normalized calls"
        )
    if (
        manifest["schema_version"] == HARDENED_SCHEMA_VERSION
        and baseline_semantics != _RAW_OUTPUT_SEMANTICS
    ):
        raise ValueError(
            "hardened manifest schema_version 2 requires versioned raw whole-output result files"
        )
    baseline_system = baseline_identity["system"]
    candidate_system = candidate_identity["system"]
    if baseline_system["name"] == candidate_system["name"]:
        raise ValueError("baseline and candidate system names must be different")
    baseline_bundle_sha256 = (
        baseline_system["bundle"]["sha256"]
        if baseline_semantics == _RAW_OUTPUT_SEMANTICS
        else baseline_system["bundle_sha256"]
    )
    candidate_bundle_sha256 = (
        candidate_system["bundle"]["sha256"]
        if candidate_semantics == _RAW_OUTPUT_SEMANTICS
        else candidate_system["bundle_sha256"]
    )
    if baseline_bundle_sha256 == candidate_bundle_sha256:
        raise ValueError("baseline and candidate bundle SHA-256 values must be different")
    if manifest["schema_version"] == HARDENED_SCHEMA_VERSION:
        permitted_rl_checkpoints = {
            item["checkpoint_sha256"]
            for item in manifest["lineage_artifacts"]
            if item["stage"] == "rl"
        }
        for system_label, system in (
            ("baseline", baseline_system),
            ("candidate", candidate_system),
        ):
            checkpoint_sha256 = system["checkpoint"]["sha256"]
            if checkpoint_sha256 not in permitted_rl_checkpoints:
                raise ValueError(
                    f"{system_label} actual checkpoint SHA-256 is absent from the frozen RL "
                    "lineage artifacts"
                )
    analysis = manifest.get("analysis")
    if not isinstance(analysis, dict):
        raise TypeError("manifest analysis is missing")
    comparison = paired_clustered_exact_action_delta_ci(
        baseline,
        candidate,
        resamples=analysis["bootstrap_resamples"],
        seed=analysis["bootstrap_seed"],
        noninferiority_margin=analysis["exact_action_noninferiority_margin"],
    )
    summary_without_hash = {
        "kind": (
            LEGACY_COMPARISON_KIND
            if baseline_semantics == _LEGACY_OUTPUT_SEMANTICS
            else (
                RAW_OUTPUT_COMPARISON_KIND
                if baseline_semantics == _RAW_OUTPUT_SEMANTICS
                else NORMALIZED_CALL_COMPARISON_KIND
            )
        ),
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest_identity,
        "slice": slice_identity,
        "baseline": baseline_identity,
        "candidate": candidate_identity,
        "exact_action_comparison": comparison,
        "promotion_decision": {
            "exact_action_noninferiority_passed": comparison["passes_noninferiority"],
            "latency_gate_evaluated": False,
            "promote": False,
            "reason": (
                "This command evaluates the exact-action gate only; the separately "
                "prespecified median-of-run p95 TTFA gate is still required."
                if baseline_semantics == _LEGACY_OUTPUT_SEMANTICS
                else (
                    "This command evaluates the exact-action gate only; the separately "
                    "prespecified median-of-run p95 TTFA gate is still required. "
                    + (
                        "The score is strict raw whole-output model evaluation."
                        if baseline_semantics == _RAW_OUTPUT_SEMANTICS
                        else (
                            "The score consumes adapter-normalized calls and is not raw "
                            "whole-output model evaluation."
                        )
                    )
                )
            ),
        },
    }
    if baseline_semantics != _LEGACY_OUTPUT_SEMANTICS:
        summary_without_hash["evaluation_input_contract"] = {
            "evaluation_semantics": baseline_semantics,
            "raw_model_output_observed": baseline_semantics == _RAW_OUTPUT_SEMANTICS,
            "strict_parse_tool_output": baseline_semantics == _RAW_OUTPUT_SEMANTICS,
            "actual_checkpoint_and_bundle_identity_verified": (
                baseline_semantics == _RAW_OUTPUT_SEMANTICS
            ),
        }
    summary_hash = _sha256(_canonical_bytes(summary_without_hash))
    payload = _canonical_bytes({**summary_without_hash, "summary_self_sha256": summary_hash})
    if args.out:
        _publish(Path(args.out), payload)
    print(json.dumps(json.loads(payload), allow_nan=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze",
        help="verify source/training identities, audit overlap, and freeze the selected slice",
    )
    freeze.add_argument("contract")
    freeze.add_argument("--slice", required=True)
    freeze.add_argument("--denylist", required=True)
    freeze.add_argument("--manifest", required=True)
    freeze.set_defaults(func=_freeze)

    verify = commands.add_parser(
        "verify",
        help="recompute a prior freeze and require byte-identical outputs",
    )
    verify.add_argument("contract")
    verify.add_argument("--slice", required=True)
    verify.add_argument("--denylist", required=True)
    verify.add_argument("--manifest", required=True)
    verify.set_defaults(func=_freeze)

    compare = commands.add_parser(
        "compare",
        help="independently AST-score paired outputs and run the frozen cluster bootstrap",
    )
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--slice", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out")
    compare.set_defaults(func=_compare)

    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"{args.command} failed: {error}") from error


if __name__ == "__main__":
    main()
