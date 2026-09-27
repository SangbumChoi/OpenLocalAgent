"""Verification and aggregation for repeated WebGPU structured-action pilot runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from localagent.train.stage_data import canonical_sha256


WEBGPU_ACTION_SUMMARY_KIND = "localagent_webgpu_structured_action_pilot_summary"
WEBGPU_ACTION_SUMMARY_SCHEMA_VERSION = 1

RUNS = 3
CASES = 20
REPETITIONS = 30
OPPORTUNITIES_PER_RUN = CASES * REPETITIONS
WARMUPS = 3
INPUT_TOKENS = 512
DEADLINES_MS = (100, 250, 500, 1000, 2000)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_LATENCY_FIELDS = (
    "harness_ttfa_ms",
    "runtime_ttfa_ms",
    "independent_validate_ms",
    "ttft_ms",
    "tpot_ms",
    "tokenize_ms",
    "inference_ms",
    "decode_control_ms",
    "dispatch_ms",
    "parse_validate_ms",
)
_FINITE_RECORD_TIMINGS = (
    "harness_ttfa_ms",
    "runtime_ttfa_ms",
    "independent_validate_ms",
    "tokenize_ms",
    "inference_ms",
    "decode_control_ms",
    "dispatch_ms",
    "parse_validate_ms",
    "ttfa_ms",
)
_DYNAMIC_METADATA = {
    "action_model_resource",
    "bundle_load_timing_ms",
    "logits_model_resource",
    "page_to_model_ready_ms",
    "warmup_records",
}
_PILOT_METADATA = {
    "benchmark_version": "rtab-0.2",
    "backend": "webgpu",
    "requested_backend": "webgpu",
    "backend_requirement": "explicit-webgpu-no-whole-session-retry",
    "benchmark_grade": True,
    "bundle_manifest_required": True,
    "policy": "structured_one_forward",
    "target_input_tokens": INPUT_TOKENS,
    "context_condition": "fixed_final_tokenizer_tokens",
    "decode_cache": None,
    "decode_strategy": "one_forward_structured_heads",
    "precision": "fp16",
    "ort_web_version": "1.27.0",
    "onnxruntime_version": "1.27.0",
    "warmups": WARMUPS,
    "repetitions": REPETITIONS,
    "cases": CASES,
    "measured_records": OPPORTUNITIES_PER_RUN,
    "case_order_seed": "slmw2026-v1",
    "concurrency": 1,
    "timer": "performance.now",
    "tab_visibility_required": True,
    "latency_clock": "harness_ttfa_ms",
}
_CORRECTED_PILOT_METADATA = _PILOT_METADATA | {
    "benchmark_version": "rtab-0.4",
    "context_condition": "fixed_compute_tokens_natural_decision_feature",
    "case_order_seed": "slmw2026-v2-trailing",
    "timeout_ms": 10_000,
    "action_timeout_ms": 10_000,
    "watchdog_scope": "every warmup and measured policy call",
    "timeout_contract": (
        "a timeout aborts the entire page collection; ORT session.run is not cancellable, "
        "so no subsequent policy call starts while timed-out inference may still be live"
    ),
    "context_padding": "single-token spaces appended after the natural assistant marker",
    "decision_feature_contract": (
        "hidden[natural_input_tokens - 1]; pointer scan bounded to natural_input_tokens"
    ),
    "raw_action_evidence_contract": (
        "each row stores normalized predicted_action, full expected_action, parse_evidence, "
        "and independent_schema including errors and the exact selected tool schema"
    ),
}
_ACTION_PROTOCOLS = {
    "rtab-0.2": _PILOT_METADATA,
    "rtab-0.4": _CORRECTED_PILOT_METADATA,
}
_ACTION_VALIDATORS = {"benchmark-json-schema-subset-v2"}


@dataclass(frozen=True)
class _Run:
    path: Path
    artifact: dict[str, int | str]
    payload: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite number at least 0.0")
    return number


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_json(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_json(item, f"{label}[{index}]")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path, root: Path) -> _Run:
    source = Path(path)
    _require(not source.is_symlink(), f"raw result must not be a symbolic link: {source}")
    _require(source.is_file(), f"raw result is missing or is not a file: {source}")
    try:
        tracked_path = source.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"raw result is outside repository root: {source}") from error
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"raw result is invalid JSON: {source}") from error
    _require(isinstance(payload, dict), f"raw result must contain an object: {source}")
    _finite_json(payload, str(source))
    return _Run(
        source,
        {
            "tracked_path": tracked_path,
            "bytes": source.stat().st_size,
            "sha256": _file_hash(source),
        },
        payload,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _latency(values: Sequence[object], label: str) -> dict[str, int | float | None]:
    finite = [_number(value, label) for value in values if value is not None]
    if not finite:
        return dict.fromkeys(("min", "mean", "p50", "p90", "p95", "p99", "max"), None) | {
            "count": 0
        }
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / len(finite),
        "p50": _percentile(finite, 0.50),
        "p90": _percentile(finite, 0.90),
        "p95": _percentile(finite, 0.95),
        "p99": _percentile(finite, 0.99),
        "max": max(finite),
    }


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(records), "at least one measured opportunity is required")
    latency = {
        field: _latency([row.get(field) for row in records], f"records.{field}")
        for field in _LATENCY_FIELDS
    }
    latency["ttfa_ms"] = copy.deepcopy(latency["harness_ttfa_ms"])
    count = len(records)
    exact = sum(row["success"] is True for row in records)
    valid = sum(row["schema_valid"] is True for row in records)
    parse_failures = sum(row["parse_failure"] is True for row in records)
    validation_failures = sum(row["validation_failure"] is True for row in records)
    total_ttfa = sum(float(row["harness_ttfa_ms"]) for row in records)
    deadlines: dict[str, Any] = {}
    for deadline in DEADLINES_MS:
        on_time = [row for row in records if float(row["harness_ttfa_ms"]) <= deadline]
        useful = [
            row
            for row in on_time
            if row["success"] is True and row["schema_valid"] is True
        ]
        deadlines[str(deadline)] = {
            "opportunities": count,
            "on_time": len(on_time),
            "on_time_rate": len(on_time) / count,
            "useful": len(useful),
            "useful_rate": len(useful) / count,
            "success_at_deadline": len(useful) / count,
            "useful_actions_per_minute": (
                len(useful) / (total_ttfa / 60_000.0) if total_ttfa else None
            ),
        }
    return {
        "latency_ms": latency,
        "exact_action_accuracy": exact / count,
        "schema_valid_rate": valid / count,
        "parse_failure_rate": parse_failures / count,
        "validation_failure_rate": validation_failures / count,
        "total_output_tokens": sum(int(row["output_tokens"]) for row in records),
        "deadline_attainment_ms": deadlines,
    }


def _validate_protocol(run: _Run, number: int) -> Mapping[str, Any]:
    payload, label = run.payload, f"run {number}"
    _require(payload.get("schema_version") == 3, f"{label} has unsupported schema")
    _require(
        payload.get("benchmark") == "localagent-held-out-action-latency",
        f"{label} has the wrong benchmark identity",
    )
    _require(bool(payload.get("created_at")), f"{label} is missing created_at")
    metadata = _mapping(payload.get("metadata"), f"{label}.metadata")
    benchmark_version = metadata.get("benchmark_version")
    expected_metadata = _ACTION_PROTOCOLS.get(str(benchmark_version))
    _require(
        expected_metadata is not None,
        f"{label}.metadata.benchmark_version is not supported",
    )
    for field, expected in expected_metadata.items():
        _require(
            metadata.get(field) == expected,
            f"{label}.metadata.{field} does not match the pilot protocol",
        )
    _require(
        metadata.get("browser") == metadata.get("user_agent"),
        f"{label} browser and user-agent identities differ",
    )
    _require(
        "Chrome/150.0.0.0" in str(metadata.get("user_agent")),
        f"{label} is not the expected Chrome 150 runtime",
    )
    provider = _mapping(metadata.get("execution_provider_request"), f"{label}.provider")
    expected_provider = {
        "requested": "webgpu",
        "session_provider_count": 1,
        "whole_session_retry": False,
        "single_provider_session_creation_succeeded": True,
    }
    for field, expected in expected_provider.items():
        _require(provider.get(field) == expected, f"{label} provider field {field} differs")
    adapter = _mapping(metadata.get("webgpu_adapter"), f"{label}.adapter")
    _require(adapter == metadata.get("gpu_adapter"), f"{label} adapter identities differ")
    _require(
        adapter.get("vendor") == "apple" and adapter.get("is_fallback_adapter") is False,
        f"{label} is not on a non-fallback Apple adapter",
    )
    expected_context_audit: dict[str, Any] = {
        "requested_input_tokens": INPUT_TOKENS,
        "verified_records": OPPORTUNITIES_PER_RUN,
        "missing_records": 0,
        "mismatched_records": 0,
    }
    if benchmark_version == "rtab-0.4":
        expected_context_audit["policy"] = "structured_one_forward"
    _require(
        metadata.get("context_audit") == expected_context_audit,
        f"{label} context audit does not cover every opportunity",
    )
    return metadata


def _validate_warmups(metadata: Mapping[str, Any], number: int) -> None:
    label = f"run {number}.metadata.warmup_records"
    warmups = _list(metadata.get("warmup_records"), label)
    _require(len(warmups) == WARMUPS, f"{label} has the wrong count")
    for index, expected_phase in enumerate(("first_inference", "warmup", "warmup")):
        row = _mapping(warmups[index], f"{label}[{index}]")
        _require(
            row.get("index") == index and row.get("phase") == expected_phase,
            f"{label}[{index}] protocol differs",
        )
        _require(
            row.get("parse_failure") is False
            and row.get("validation_failure") is False,
            f"{label}[{index}] failed",
        )
        if metadata["benchmark_version"] == "rtab-0.4":
            _validate_v04_action_evidence(row, f"{label}[{index}]")
        harness = _number(row.get("harness_ttfa_ms"), f"{label}[{index}].harness_ttfa_ms")
        legacy = _number(row.get("ttfa_ms"), f"{label}[{index}].ttfa_ms")
        _require(harness == legacy, f"{label}[{index}] TTFA alias differs")
    for field, value in _mapping(
        metadata.get("bundle_load_timing_ms"), f"run {number}.bundle_load_timing_ms"
    ).items():
        _number(value, f"run {number}.bundle_load_timing_ms.{field}")
    _number(metadata.get("page_to_model_ready_ms"), f"run {number}.page_to_model_ready_ms")


def _asset(
    metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    evidence_key: str,
    hash_key: str,
    filename: str,
    label: str,
) -> dict[str, int | str]:
    evidence = _mapping(
        _mapping(metadata.get("runtime_asset_evidence"), f"{label}.runtime_assets").get(
            evidence_key
        ),
        f"{label}.{evidence_key}",
    )
    manifest = _mapping(artifacts.get(filename), f"{label}.manifest.{filename}")
    digest = _sha256(metadata.get(hash_key), f"{label}.{hash_key}")
    size = _integer(evidence.get("bytes"), f"{label}.{filename}.bytes", 1)
    _require(
        evidence.get("file") == filename
        and evidence.get("sha256") == digest
        and evidence.get("manifest_sha256") == digest
        and manifest.get("sha256") == digest,
        f"{label} {filename} digest binding differs",
    )
    _require(
        evidence.get("manifest_verified") is True
        and evidence.get("manifest_bytes") == size
        and manifest.get("bytes") == size,
        f"{label} {filename} byte binding differs",
    )
    return {"file": filename, "bytes": size, "sha256": digest}


def _validate_identity(metadata: Mapping[str, Any], number: int) -> dict[str, Any]:
    label = f"run {number}.metadata"
    manifest = _mapping(metadata.get("bundle_manifest"), f"{label}.bundle_manifest")
    artifacts = _mapping(manifest.get("artifacts"), f"{label}.manifest.artifacts")
    _require(manifest.get("schema_version") == 3, f"{label} manifest schema differs")
    checkpoint = _sha256(metadata.get("checkpoint_hash"), f"{label}.checkpoint_hash")
    _require(
        manifest.get("checkpoint_sha256") == checkpoint,
        f"{label} checkpoint and manifest identities differ",
    )
    _require(manifest.get("checkpoint_stage") == "sft", f"{label} is not an SFT checkpoint")

    graph_file = metadata.get("policy_model_url")
    _require(
        graph_file
        == metadata.get("model_url")
        == metadata.get("action_model_url")
        == "action_model.fp16.onnx",
        f"{label} action graph selection differs",
    )
    graph_manifest = _mapping(artifacts.get(graph_file), f"{label}.manifest.{graph_file}")
    graph_evidence = _mapping(metadata.get("model_byte_evidence"), f"{label}.model_bytes")
    graph_hash = _sha256(metadata.get("graph_hash"), f"{label}.graph_hash")
    graph_bytes = _integer(metadata.get("model_bytes"), f"{label}.model_bytes", 1)
    _require(
        all(
            value == graph_hash
            for value in (
                metadata.get("model_hash"),
                metadata.get("manifest_graph_hash"),
                graph_manifest.get("sha256"),
                graph_evidence.get("sha256"),
                graph_evidence.get("manifest_sha256"),
            )
        ),
        f"{label} graph identities differ",
    )
    _require(
        all(
            value == graph_bytes
            for value in (
                metadata.get("manifest_model_bytes"),
                graph_manifest.get("bytes"),
                graph_evidence.get("bytes"),
                graph_evidence.get("manifest_bytes"),
            )
        )
        and graph_evidence.get("manifest_verified") is True
        and graph_evidence.get("session_source") == "in_memory_verified_bytes",
        f"{label} graph byte binding differs",
    )

    asset_specs = (
        ("heads_json", "heads_hash", "heads.json"),
        ("dispatch_heads_json", "dispatch_heads_hash", "dispatch_heads.json"),
        ("tokenizer", "tokenizer_hash", "tokenizer.json"),
        ("meta_json", "meta_file_hash", "meta.json"),
    )
    bound = {
        key: _asset(metadata, artifacts, evidence, digest, filename, label)
        for key, (evidence, digest, filename) in zip(
            ("heads", "dispatch_heads", "tokenizer", "model_metadata"),
            asset_specs,
            strict=True,
        )
    }
    _require(
        metadata.get("manifest_tokenizer_hash") == bound["tokenizer"]["sha256"],
        f"{label} tokenizer and manifest identities differ",
    )

    bundle_evidence = _mapping(
        metadata.get("bundle_manifest_byte_evidence"), f"{label}.manifest_bytes"
    )
    _require(
        bundle_evidence.get("role") == "parsed_bundle_manifest_trust_anchor"
        and bundle_evidence.get("external_expected_identity") is None
        and bundle_evidence.get("manifest_verified") is False,
        f"{label} overstates independent manifest verification",
    )
    suite_evidence = _mapping(metadata.get("suite_byte_evidence"), f"{label}.suite")
    suite_hash = _sha256(metadata.get("suite_sha256"), f"{label}.suite_sha256")
    suite_bytes = _integer(metadata.get("suite_bytes"), f"{label}.suite_bytes", 1)
    _require(
        suite_evidence.get("sha256")
        == suite_evidence.get("expected_sha256")
        == suite_hash
        and suite_evidence.get("bytes")
        == suite_evidence.get("expected_bytes")
        == suite_bytes
        and suite_evidence.get("identity_verified") is True,
        f"{label} held-out suite identity differs",
    )
    parity = _mapping(manifest.get("parity_gate"), f"{label}.parity_gate")
    _require(
        parity.get("hard_gate") is True and parity.get("passed") is True,
        f"{label} export parity gate did not pass",
    )
    parameters = _integer(manifest.get("model_parameters"), f"{label}.parameters", 1)
    _require(metadata.get("model_parameters") == parameters, f"{label} parameters differ")

    bound["model_metadata"]["canonical_sha256"] = _sha256(
        metadata.get("model_meta_canonical_sha256"), f"{label}.model_meta_canonical_sha256"
    )
    return {
        "checkpoint": {
            "sha256": checkpoint,
            "stage": manifest["checkpoint_stage"],
            "step": _integer(manifest.get("checkpoint_step"), f"{label}.checkpoint_step"),
        },
        "model_config": {
            "name": manifest["config_name"],
            "canonical_sha256": _sha256(
                manifest.get("model_config_sha256"), f"{label}.model_config_sha256"
            ),
            "parameters": parameters,
        },
        "graph": {"file": graph_file, "bytes": graph_bytes, "sha256": graph_hash},
        "bundle_manifest": {
            "raw_bytes": _integer(bundle_evidence.get("bytes"), f"{label}.manifest.bytes", 1),
            "raw_sha256": _sha256(
                bundle_evidence.get("sha256"), f"{label}.manifest.sha256"
            ),
            "canonical_sha256": _sha256(
                metadata.get("bundle_manifest_canonical_sha256"),
                f"{label}.manifest.canonical_sha256",
            ),
            "schema_version": manifest["schema_version"],
        },
        **bound,
        "held_out_suite": {
            "bytes": suite_bytes,
            "sha256": suite_hash,
            "schema_version": metadata["suite_schema_version"],
        },
    }


def _canonical_equal(left: object, right: object) -> bool:
    return json.dumps(
        left,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        right,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalized_action(value: object, label: str) -> Mapping[str, Any]:
    action = _mapping(value, label)
    if action.get("abstain") is True:
        _require(set(action) == {"abstain"}, f"{label} is not a normalized abstention")
    else:
        _require(set(action) == {"tool", "args"}, f"{label} is not a normalized tool action")
        _require(
            action.get("tool") is None or isinstance(action.get("tool"), str),
            f"{label}.tool must be a string or null",
        )
    return action


def _schema_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _schema_type_matches(value: object, expected: object) -> bool:
    if expected is None:
        return True
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if expected == "number":
        return _schema_number(value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _schema_limit(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _value_schema_errors(
    value: object,
    schema: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_canonical_equal(candidate, value) for candidate in enum):
        errors.append(f"{path} is not in the declared enum.")
    if "const" in schema and not _canonical_equal(value, schema["const"]):
        errors.append(f"{path} does not equal the declared constant.")

    expected_type = schema.get("type")
    if not _schema_type_matches(value, expected_type):
        errors.append(f"{path} does not have JSON Schema type {expected_type}.")
        return

    if expected_type == "object":
        object_value = _mapping(value, path)
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        for name in required:
            if name not in object_value:
                errors.append(f"{path}.{name} is required.")
        for name, child in object_value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                _value_schema_errors(child, child_schema, f"{path}.{name}", errors)
            elif schema.get("additionalProperties") is not True:
                errors.append(f"{path}.{name} is not declared by the tool schema.")
    if expected_type == "array":
        array_value = _list(value, path)
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and not isinstance(minimum, bool) and len(array_value) < minimum:
            errors.append(f"{path} has fewer than {minimum} items.")
        if isinstance(maximum, int) and not isinstance(maximum, bool) and len(array_value) > maximum:
            errors.append(f"{path} has more than {maximum} items.")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, child in enumerate(array_value):
                _value_schema_errors(child, item_schema, f"{path}[{index}]", errors)
    if expected_type == "string":
        string_value = str(value)
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and not isinstance(minimum, bool) and len(string_value) < minimum:
            errors.append(f"{path} is shorter than minLength {minimum}.")
        if isinstance(maximum, int) and not isinstance(maximum, bool) and len(string_value) > maximum:
            errors.append(f"{path} is longer than maxLength {maximum}.")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, string_value) is None:
                    errors.append(f"{path} does not match the declared pattern.")
            except re.error:
                errors.append(f"{path} has an invalid schema pattern.")
    if expected_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if _schema_number(minimum) and float(value) < float(minimum):
            errors.append(f"{path} is below minimum {_schema_limit(minimum)}.")
        if _schema_number(maximum) and float(value) > float(maximum):
            errors.append(f"{path} is above maximum {_schema_limit(maximum)}.")


def _schema_result(
    action: Mapping[str, Any],
    stored: object,
    label: str,
) -> Mapping[str, Any]:
    result = _mapping(stored, label)
    _require(result.get("validator") in _ACTION_VALIDATORS, f"{label}.validator differs")
    errors = _list(result.get("errors"), f"{label}.errors")
    _require(all(isinstance(error, str) for error in errors), f"{label}.errors must be strings")

    if action.get("abstain") is True:
        recomputed_errors: list[str] = []
        schema_tool = None
        tool_schema = None
    else:
        schema_tool = result.get("schema_tool")
        tool_schema_value = result.get("tool_schema")
        if tool_schema_value is None:
            schema_tool = None
            tool_schema = None
            recomputed_errors = [
                "Unknown tool "
                + json.dumps(action.get("tool"), ensure_ascii=False, separators=(",", ":"))
                + "."
            ]
        else:
            tool_schema = _mapping(tool_schema_value, f"{label}.tool_schema")
            _require(
                isinstance(schema_tool, str) and schema_tool == action.get("tool"),
                f"{label}.schema_tool does not bind the predicted tool",
            )
            recomputed_errors = []
            _value_schema_errors(action.get("args"), tool_schema, "$.args", recomputed_errors)

    expected = {
        "validator": result["validator"],
        "valid": len(recomputed_errors) == 0,
        "errors": recomputed_errors,
        "schema_tool": schema_tool,
        "tool_schema": tool_schema,
    }
    _require(dict(result) == expected, f"{label} does not reproduce from raw action and schema")
    return result


def _action_scores(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, bool]:
    if expected.get("abstain") is True:
        exact = predicted.get("abstain") is True
        return {"exact_tool": exact, "exact_args": exact, "exact_action": exact}
    exact_tool = (
        predicted.get("abstain") is not True
        and predicted.get("tool") == expected.get("tool")
    )
    exact_args = exact_tool and _canonical_equal(predicted.get("args"), expected.get("args"))
    return {
        "exact_tool": exact_tool,
        "exact_args": exact_args,
        "exact_action": exact_tool and exact_args,
    }


def _validate_v04_action_evidence(row: Mapping[str, Any], label: str) -> None:
    predicted = _normalized_action(row.get("predicted_action"), f"{label}.predicted_action")
    expected = _normalized_action(row.get("expected_action"), f"{label}.expected_action")
    schema = _schema_result(predicted, row.get("independent_schema"), f"{label}.independent_schema")
    parse = _mapping(row.get("parse_evidence"), f"{label}.parse_evidence")
    _require(
        parse.get("policy") == "structured_one_forward"
        and parse.get("inference_passes") == 1
        and parse.get("parse_kind") == "structured_heads",
        f"{label}.parse_evidence is not one structured inference pass",
    )
    for field in ("parse_failure", "runtime_validation_failure"):
        _require(isinstance(parse.get(field), bool), f"{label}.parse_evidence.{field} differs")
    scores = _action_scores(predicted, expected)
    for field, recomputed in scores.items():
        _require(row.get(field) is recomputed, f"{label}.{field} disagrees with raw actions")
    _require(
        row.get("success") is scores["exact_action"],
        f"{label}.success disagrees with raw actions",
    )
    _require(
        row.get("schema_valid") is schema["valid"],
        f"{label}.schema_valid disagrees with independent schema evidence",
    )
    _require(
        row.get("validation_failure") is (not schema["valid"]),
        f"{label}.validation_failure disagrees with independent schema evidence",
    )
    _require(
        row.get("parse_failure") is parse["parse_failure"],
        f"{label}.parse_failure disagrees with parse evidence",
    )
    predicted_tool = None if predicted.get("abstain") is True else predicted.get("tool")
    expected_tool = None if expected.get("abstain") is True else expected.get("tool")
    _require(
        row.get("predicted_tool") == predicted_tool
        and row.get("expected_tool") == expected_tool,
        f"{label} tool aliases disagree with normalized actions",
    )
    _require(
        row.get("action_timeout_ms") == 10_000
        and row.get("watchdog_outcome") == "completed_before_timeout",
        f"{label} watchdog evidence differs",
    )


def _signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[field]
        for field in (
            "case_id",
            "family",
            "repetition",
            "order_index",
            "expected_tool",
            "input_bytes",
            "natural_input_tokens",
            "context_padding_tokens",
        )
    )


def _validate_records(
    metadata: Mapping[str, Any], value: object, number: int
) -> tuple[list[Mapping[str, Any]], list[tuple[Any, ...]]]:
    label = f"run {number}.records"
    values = _list(value, label)
    _require(
        len(values) == OPPORTUNITIES_PER_RUN,
        f"{label} opportunity count must be exactly {OPPORTUNITIES_PER_RUN}",
    )
    records = [_mapping(value, f"{label}[{index}]") for index, value in enumerate(values)]
    benchmark_version = metadata["benchmark_version"]
    for index, row in enumerate(records):
        row_label = f"{label}[{index}]"
        _require(
            row.get("order_index") == index % CASES
            and row.get("repetition") == index // CASES,
            f"{row_label} case order differs",
        )
        _require(
            row.get("backend") == metadata["backend"]
            and row.get("policy") == metadata["policy"]
            and row.get("input_tokens") == INPUT_TOKENS,
            f"{row_label} protocol differs",
        )
        natural = _integer(row.get("natural_input_tokens"), f"{row_label}.natural")
        padding = _integer(row.get("context_padding_tokens"), f"{row_label}.padding")
        _require(natural + padding == INPUT_TOKENS, f"{row_label} padding differs")
        if benchmark_version == "rtab-0.4":
            _require(
                row.get("context_padding_placement") == "after_natural_assistant_marker"
                and row.get("decision_input_tokens") == natural
                and row.get("decision_feature_index") == natural - 1,
                f"{row_label} corrected decision-feature contract differs",
            )
        _require(
            row.get("decode_strategy") == metadata["decode_strategy"]
            and row.get("decode_cache") is None
            and row.get("inference_passes") == 1
            and row.get("parse_kind") == "structured_heads",
            f"{row_label} execution strategy differs",
        )
        for field in _FINITE_RECORD_TIMINGS:
            _number(row.get(field), f"{row_label}.{field}")
        _require(
            row["harness_ttfa_ms"] == row["ttfa_ms"],
            f"{row_label} TTFA alias differs",
        )
        for field in ("success", "schema_valid", "parse_failure", "validation_failure"):
            _require(isinstance(row.get(field), bool), f"{row_label}.{field} differs")
        _integer(row.get("output_tokens"), f"{row_label}.output_tokens")
        if benchmark_version == "rtab-0.4":
            _validate_v04_action_evidence(row, row_label)

    case_counts = Counter(str(row.get("case_id")) for row in records)
    _require(
        len(case_counts) == CASES and set(case_counts.values()) == {REPETITIONS},
        f"{label} must contain {CASES} cases exactly {REPETITIONS} times each",
    )
    opportunities = Counter((row["case_id"], row["repetition"]) for row in records)
    _require(
        len(opportunities) == OPPORTUNITIES_PER_RUN
        and set(opportunities.values()) == {1},
        f"{label} opportunities are not unique and complete",
    )
    return records, [_signature(row) for row in records]


def _static(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in metadata.items()
        if key not in _DYNAMIC_METADATA
    }


def _run_level(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "unit_of_replication": "independent browser page/session run",
        "reported_point_estimate": "median of the three within-run percentiles",
        "reported_range": "minimum and maximum of the three within-run percentiles",
    }
    for percentile in ("p50", "p95", "p99"):
        values = [
            float(summary["latency_ms"]["harness_ttfa_ms"][percentile])
            for summary in summaries
        ]
        output[percentile] = {
            "by_run": values,
            "median": median(values),
            "range": [min(values), max(values)],
        }
    return output


def _outcomes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_abstain = [row for row in records if row.get("expected_tool") is None]
    expected_tool = [row for row in records if row.get("expected_tool") is not None]
    exact_abstain = sum(row["success"] is True for row in expected_abstain)
    exact_tool = sum(row["success"] is True for row in expected_tool)
    return {
        "expected_abstention_opportunities": len(expected_abstain),
        "expected_tool_opportunities": len(expected_tool),
        "null_predicted_tool": sum(row.get("predicted_tool") is None for row in records),
        "non_null_predicted_tool": sum(
            row.get("predicted_tool") is not None for row in records
        ),
        "exact_expected_abstentions": exact_abstain,
        "exact_expected_tools": exact_tool,
        "exact_rate_expected_abstentions": exact_abstain / len(expected_abstain),
        "exact_rate_expected_tools": exact_tool / len(expected_tool),
    }


def _outcome_evidence(benchmark_version: str) -> dict[str, str]:
    if benchmark_version == "rtab-0.4":
        return {
            "mode": "independently_recomputed",
            "exact_action": (
                "recomputed from normalized predicted_action and full expected_action; all "
                "stored exact-tool, exact-args, exact-action, and success booleans must agree"
            ),
            "schema_validity": (
                "recomputed from predicted_action and independent_schema.tool_schema; stored "
                "validator result, errors, schema-valid, and validation-failure fields must agree"
            ),
        }
    return {
        "mode": "browser_reported_non_recomputable",
        "exact_action": (
            "historical rtab-0.2 rows omit predicted args and full expected actions, so exact "
            "action outcomes are browser-reported and cannot be independently recomputed"
        ),
        "schema_validity": (
            "historical rtab-0.2 rows omit validator errors and selected tool schemas, so schema "
            "validity is browser-reported and cannot be independently recomputed"
        ),
    }


def build_webgpu_action_summary(
    raw_paths: Sequence[str | Path], *, repository_root: str | Path
) -> dict[str, Any]:
    """Validate exactly three pilot runs and build their deterministic aggregate."""

    _require(len(raw_paths) == RUNS, f"expected exactly {RUNS} independent browser runs")
    loaded = [_load(path, Path(repository_root)) for path in raw_paths]
    _require(
        len({run.artifact["sha256"] for run in loaded}) == RUNS,
        "raw browser runs must have distinct file identities",
    )
    metadata: list[Mapping[str, Any]] = []
    identities: list[dict[str, Any]] = []
    records: list[list[Mapping[str, Any]]] = []
    signatures: list[list[tuple[Any, ...]]] = []
    summaries: list[dict[str, Any]] = []
    for number, run in enumerate(loaded, start=1):
        run_metadata = _validate_protocol(run, number)
        _validate_warmups(run_metadata, number)
        run_identity = _validate_identity(run_metadata, number)
        run_records, signature = _validate_records(
            run_metadata, run.payload.get("records"), number
        )
        summary = _summarize(run_records)
        _require(
            run.payload.get("summary") == summary,
            f"run {number} embedded summary does not reproduce from records",
        )
        metadata.append(run_metadata)
        identities.append(run_identity)
        records.append(run_records)
        signatures.append(signature)
        summaries.append(summary)

    for index in range(1, RUNS):
        _require(
            _static(metadata[index]) == _static(metadata[0]),
            f"run {index + 1} protocol or runtime identity differs from run 1",
        )
        _require(
            identities[index] == identities[0],
            f"run {index + 1} artifact identity differs from run 1",
        )
        _require(
            signatures[index] == signatures[0],
            f"run {index + 1} opportunity order differs from run 1",
        )
    created_at = [str(run.payload["created_at"]) for run in loaded]
    _require(
        created_at == sorted(created_at) and len(set(created_at)) == RUNS,
        "raw browser runs must be supplied in distinct chronological order",
    )

    run_rows = []
    for number, (run, summary, run_records) in enumerate(
        zip(loaded, summaries, records, strict=True), start=1
    ):
        run_rows.append(
            {
                "run": number,
                "created_at": run.payload["created_at"],
                "raw_artifact": copy.deepcopy(run.artifact),
                "opportunities": len(run_records),
                "exact_actions": sum(row["success"] is True for row in run_records),
                "schema_valid_actions": sum(
                    row["schema_valid"] is True for row in run_records
                ),
                "ttfa_ms": copy.deepcopy(summary["latency_ms"]["harness_ttfa_ms"]),
                "exact_action_accuracy": summary["exact_action_accuracy"],
                "schema_valid_rate": summary["schema_valid_rate"],
                "parse_failure_rate": summary["parse_failure_rate"],
                "validation_failure_rate": summary["validation_failure_rate"],
                "success_at_deadline_ms": copy.deepcopy(
                    summary["deadline_attainment_ms"]
                ),
            }
        )

    first = metadata[0]
    pooled = [row for run_records in records for row in run_records]
    report: dict[str, Any] = {
        "kind": WEBGPU_ACTION_SUMMARY_KIND,
        "schema_version": WEBGPU_ACTION_SUMMARY_SCHEMA_VERSION,
        "protocol": {
            "raw_schema_version": loaded[0].payload["schema_version"],
            "benchmark": loaded[0].payload["benchmark"],
            "benchmark_version": first["benchmark_version"],
            "backend": first["backend"],
            "backend_requirement": first["backend_requirement"],
            "execution_provider_request": copy.deepcopy(
                first["execution_provider_request"]
            ),
            "onnxruntime_web_version": first["onnxruntime_version"],
            "browser_user_agent": first["user_agent"],
            "gpu_adapter": copy.deepcopy(first["webgpu_adapter"]),
            "policy": first["policy"],
            "decode_strategy": first["decode_strategy"],
            "precision": first["precision"],
            "target_input_tokens": first["target_input_tokens"],
            "warmups_per_run": first["warmups"],
            "cases": first["cases"],
            "repetitions_per_case": first["repetitions"],
            "opportunities_per_run": first["measured_records"],
            "case_order_seed": first["case_order_seed"],
            "concurrency": first["concurrency"],
            "latency_clock": first["latency_clock"],
            "latency_boundaries": copy.deepcopy(first["latency_boundaries"]),
            "deadlines_ms": list(DEADLINES_MS),
            "record_outcome_evidence": _outcome_evidence(first["benchmark_version"]),
        },
        "identity": copy.deepcopy(identities[0]),
        "validation": {
            "status": "mechanically_valid",
            "checks": {
                "three_distinct_chronological_runs": True,
                "protocol_and_runtime_identity_consistent": True,
                "checkpoint_graph_and_manifest_identity_consistent": True,
                "runtime_assets_manifest_verified": True,
                "export_parity_gate_passed": True,
                "held_out_suite_identity_verified": True,
                "exact_opportunity_accounting": True,
                "measured_opportunity_order_consistent": True,
                "finite_non_negative_ttfa": True,
                "embedded_summaries_reproduced": True,
                "record_outcome_evidence_scope_labeled": True,
            },
        },
        "runs": run_rows,
        "aggregate": {
            "run_count": RUNS,
            "opportunities": len(pooled),
            "exact_actions": sum(row["success"] is True for row in pooled),
            "schema_valid_actions": sum(row["schema_valid"] is True for row in pooled),
            "parse_failures": sum(row["parse_failure"] is True for row in pooled),
            "validation_failures": sum(
                row["validation_failure"] is True for row in pooled
            ),
            "outcome_breakdown": _outcomes(pooled),
            "run_level_ttfa_ms": _run_level(summaries),
            "pooled_measurement_unit": "measured action opportunity",
            "pooled_metrics": _summarize(pooled),
        },
    }
    if first["benchmark_version"] == "rtab-0.4":
        report["protocol"].update(
            {
                "context_condition": first["context_condition"],
                "context_padding": first["context_padding"],
                "decision_feature_contract": first["decision_feature_contract"],
                "action_timeout_ms": first["action_timeout_ms"],
                "timeout_contract": first["timeout_contract"],
            }
        )
    report["summary_sha256"] = canonical_sha256(report)
    return report


def write_webgpu_action_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a finite, sorted summary after checking its self-hash."""

    payload = copy.deepcopy(dict(summary))
    recorded_hash = payload.pop("summary_sha256", None)
    _require(
        recorded_hash == canonical_sha256(payload),
        "WebGPU action summary self-hash is missing or invalid",
    )
    payload["summary_sha256"] = recorded_hash
    _finite_json(payload, "WebGPU action summary")
    encoded = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
