"""Audited aggregation for repeated WebGPU single-step DOM pilot runs."""

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


WEBGPU_DOM_SUMMARY_KIND = "localagent_webgpu_single_step_dom_pilot_summary"
WEBGPU_DOM_SUMMARY_SCHEMA_VERSION = 1

RUNS = 3
CASES = 8
REPETITIONS = 30
RECORDS_PER_RUN = CASES * REPETITIONS
WARMUPS = 3
INPUT_TOKENS = 512
DEADLINES_MS = (100, 250, 500, 1000, 2000)
SUPPORTED_TOOLS = (
    "click",
    "double_click",
    "type_text",
    "key_press",
    "scroll",
    "drag",
    "move_cursor",
    "open_url",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_LATENCY_FIELDS = (
    "harness_ttfa_ms",
    "runtime_ttfa_ms",
    "independent_validate_ms",
    "model_wall_ms",
    "tool_ms",
    "paint_wait_ms",
    "closed_loop_ms",
)
_SCORE_FIELDS = (
    "exact_tool",
    "exact_args",
    "exact_action",
    "schema_valid",
    "final_dom_valid",
    "state_transition",
    "closed_loop_success",
)
_DYNAMIC_METADATA = {
    "bundle_load_timing_ms",
    "model_resource",
    "page_to_model_ready_ms",
}
_PILOT_METADATA = {
    "benchmark_version": "rtab-dom-0.2",
    "backend": "webgpu",
    "requested_backend": "webgpu",
    "backend_requirement": "explicit-webgpu-no-whole-session-retry",
    "benchmark_grade": True,
    "bundle_manifest_required": True,
    "model_url": "action_model.fp16.onnx",
    "precision": "fp16",
    "ort_web_version": "1.27.0",
    "onnxruntime_version": "1.27.0",
    "target_input_tokens": INPUT_TOKENS,
    "context_condition": "fixed_final_tokenizer_tokens",
    "suite_schema_version": 1,
    "suite_expected_actions_schema_validated": True,
    "case_order_seed": "dom-loop-v1",
    "warmups": WARMUPS,
    "repetitions": REPETITIONS,
    "cases": CASES,
    "measured_records": RECORDS_PER_RUN,
    "concurrency": 1,
    "timer": "performance.now",
    "paint_barrier": "two consecutive requestAnimationFrame callbacks",
    "latency_clock": "harness_ttfa_ms",
}
_CORRECTED_PILOT_METADATA = _PILOT_METADATA | {
    "benchmark_version": "rtab-dom-0.4",
    "context_condition": "fixed_compute_tokens_natural_decision_feature",
    "case_order_seed": "dom-loop-v2-trailing",
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
_DOM_PROTOCOLS = {
    "rtab-dom-0.2": _PILOT_METADATA,
    "rtab-dom-0.4": _CORRECTED_PILOT_METADATA,
}
_DOM_VALIDATORS = {
    "browser-task-json-schema-subset-v2",
}
_EXPECTED_IDENTITIES = {
    "checkpoint": "79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1",
    "graph": "b91e7f84077155640a5e288a7c58c2245c298859ddd86bd7268e71039e65c49a",
    "tokenizer": "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c",
    "suite": "4c46b5b347257b81e716ec0a20a6c6116df716466e1ba8e8a74a117bb5708971",
    "bundle_manifest_raw": (
        "86bbee00d783ca69af02843a4cf935ff978612b81b6a2fedd47fd943e611bee4"
    ),
    "bundle_manifest_canonical": (
        "5fee08dfaf4dab4a4d58f506c3fe55ba38c7168ea929f316f307351db7be3fd5"
    ),
}


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
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
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


def _rate(records: Sequence[Mapping[str, Any]], score: str) -> float:
    return (
        sum(_mapping(record["score"], "record.score").get(score) is True for record in records)
        / len(records)
        if records
        else 0.0
    )


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(records), "at least one measured DOM record is required")
    latency = {
        key: _latency(
            [_mapping(record["latency_ms"], "record.latency_ms").get(key) for record in records],
            f"records.latency_ms.{key}",
        )
        for key in _LATENCY_FIELDS
    }
    by_action: dict[str, Any] = {}
    for tool in SUPPORTED_TOOLS:
        matching = [
            record
            for record in records
            if _mapping(record["expected"], "record.expected").get("tool") == tool
        ]
        if matching:
            by_action[tool] = {
                "count": len(matching),
                "exact_action_rate": _rate(matching, "exact_action"),
                "schema_valid_rate": _rate(matching, "schema_valid"),
                "final_dom_rate": _rate(matching, "final_dom_valid"),
                "closed_loop_success_rate": _rate(matching, "closed_loop_success"),
                "closed_loop_ms": _latency(
                    [
                        _mapping(record["latency_ms"], "record.latency_ms")[
                            "closed_loop_ms"
                        ]
                        for record in matching
                    ],
                    f"records.{tool}.closed_loop_ms",
                ),
            }

    harness = [
        float(_mapping(record["latency_ms"], "record.latency_ms")["harness_ttfa_ms"])
        for record in records
    ]
    total_harness = sum(harness)
    deadlines: dict[str, Any] = {}
    for deadline in DEADLINES_MS:
        on_time = [
            record
            for record in records
            if float(_mapping(record["latency_ms"], "record.latency_ms")["harness_ttfa_ms"])
            <= deadline
        ]
        useful = [
            record
            for record in on_time
            if _mapping(record["score"], "record.score").get("exact_action") is True
            and _mapping(record["score"], "record.score").get("schema_valid") is True
        ]
        deadlines[str(deadline)] = {
            "opportunities": len(records),
            "on_time": len(on_time),
            "on_time_rate": len(on_time) / len(records),
            "useful": len(useful),
            "success_at_deadline": len(useful) / len(records),
            "useful_actions_per_minute": (
                len(useful) / (total_harness / 60_000.0) if total_harness else None
            ),
        }

    return {
        "records": len(records),
        "exact_tool_rate": _rate(records, "exact_tool"),
        "exact_args_rate": _rate(records, "exact_args"),
        "exact_action_rate": _rate(records, "exact_action"),
        "schema_valid_rate": _rate(records, "schema_valid"),
        "final_dom_rate": _rate(records, "final_dom_valid"),
        "state_transition_rate": _rate(records, "state_transition"),
        "closed_loop_success_rate": _rate(records, "closed_loop_success"),
        "latency_ms": latency,
        "deadline_attainment_ms": deadlines,
        "by_action": by_action,
    }


def _validate_protocol(run: _Run, number: int) -> Mapping[str, Any]:
    payload, label = run.payload, f"run {number}"
    _require(payload.get("schema_version") == 2, f"{label} has unsupported schema")
    _require(
        payload.get("benchmark") == "localagent-single-step-dom-microtasks",
        f"{label} has the wrong benchmark identity",
    )
    _require(bool(payload.get("created_at")), f"{label} is missing created_at")
    metadata = _mapping(payload.get("metadata"), f"{label}.metadata")
    benchmark_version = metadata.get("benchmark_version")
    expected_metadata = _DOM_PROTOCOLS.get(str(benchmark_version))
    _require(
        expected_metadata is not None,
        f"{label}.metadata.benchmark_version is not supported",
    )
    for field, expected in expected_metadata.items():
        _require(
            metadata.get(field) == expected,
            f"{label}.metadata.{field} does not match the DOM pilot protocol",
        )
    provider = _mapping(metadata.get("execution_provider_request"), f"{label}.provider")
    for field, expected in {
        "requested": "webgpu",
        "session_provider_count": 1,
        "whole_session_retry": False,
        "single_provider_session_creation_succeeded": True,
    }.items():
        _require(provider.get(field) == expected, f"{label} provider field {field} differs")
    _require(
        metadata.get("browser") == metadata.get("user_agent")
        and "Chrome/150.0.0.0" in str(metadata.get("user_agent")),
        f"{label} browser identity differs",
    )
    adapter = _mapping(metadata.get("webgpu_adapter"), f"{label}.adapter")
    _require(
        adapter == metadata.get("gpu_adapter")
        and adapter.get("vendor") == "apple"
        and adapter.get("is_fallback_adapter") is False,
        f"{label} adapter identity differs",
    )
    _require(
        metadata.get("context_audit")
        == {
            "requested_input_tokens": INPUT_TOKENS,
            "verified_records": RECORDS_PER_RUN,
            "missing_records": 0,
            "mismatched_records": 0,
        },
        f"{label} context audit does not cover every record",
    )
    _require(
        metadata.get("fixture_contract")
        == {"name": "localagent-dom-fixtures", "version": 1},
        f"{label} fixture contract differs",
    )
    return metadata


def _asset(
    metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    evidence_key: str,
    hash_key: str,
    filename: str,
    label: str,
) -> dict[str, int | str]:
    runtime = _mapping(metadata.get("runtime_asset_evidence"), f"{label}.runtime_assets")
    evidence = _mapping(runtime.get(evidence_key), f"{label}.{evidence_key}")
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
    checkpoint = _sha256(metadata.get("checkpoint_hash"), f"{label}.checkpoint_hash")
    graph = _sha256(metadata.get("graph_hash"), f"{label}.graph_hash")
    tokenizer = _sha256(metadata.get("tokenizer_hash"), f"{label}.tokenizer_hash")
    suite = _sha256(metadata.get("suite_sha256"), f"{label}.suite_sha256")
    _require(
        checkpoint == _EXPECTED_IDENTITIES["checkpoint"],
        f"{label} checkpoint does not match the pilot identity",
    )
    _require(
        graph == _EXPECTED_IDENTITIES["graph"],
        f"{label} graph does not match the pilot identity",
    )
    _require(
        tokenizer == _EXPECTED_IDENTITIES["tokenizer"],
        f"{label} tokenizer does not match the pilot identity",
    )
    _require(
        suite == _EXPECTED_IDENTITIES["suite"],
        f"{label} suite does not match the pilot identity",
    )
    _require(
        manifest.get("schema_version") == 3
        and manifest.get("checkpoint_sha256") == checkpoint
        and manifest.get("checkpoint_stage") == "sft",
        f"{label} checkpoint and manifest identities differ",
    )

    graph_file = metadata["model_url"]
    graph_manifest = _mapping(artifacts.get(graph_file), f"{label}.manifest.{graph_file}")
    graph_evidence = _mapping(metadata.get("model_byte_evidence"), f"{label}.model_bytes")
    graph_bytes = _integer(metadata.get("model_bytes"), f"{label}.model_bytes", 1)
    _require(
        all(
            value == graph
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
        ("heads", "heads_json", "heads_hash", "heads.json"),
        (
            "dispatch_heads",
            "dispatch_heads_json",
            "dispatch_heads_hash",
            "dispatch_heads.json",
        ),
        ("tokenizer", "tokenizer", "tokenizer_hash", "tokenizer.json"),
        ("model_metadata", "meta_json", "meta_file_hash", "meta.json"),
    )
    bound = {
        key: _asset(metadata, artifacts, evidence, hash_key, filename, label)
        for key, evidence, hash_key, filename in asset_specs
    }
    _require(
        bound["tokenizer"]["sha256"] == tokenizer
        and metadata.get("manifest_tokenizer_hash") == tokenizer,
        f"{label} tokenizer binding differs",
    )

    suite_evidence = _mapping(metadata.get("suite_byte_evidence"), f"{label}.suite")
    suite_bytes = _integer(metadata.get("suite_bytes"), f"{label}.suite_bytes", 1)
    _require(
        suite_evidence.get("sha256")
        == suite_evidence.get("expected_sha256")
        == suite
        and suite_evidence.get("bytes")
        == suite_evidence.get("expected_bytes")
        == suite_bytes
        and suite_evidence.get("identity_verified") is True,
        f"{label} suite byte binding differs",
    )
    bundle = _mapping(
        metadata.get("bundle_manifest_byte_evidence"), f"{label}.manifest_bytes"
    )
    raw_manifest = _sha256(bundle.get("sha256"), f"{label}.manifest.sha256")
    canonical_manifest = _sha256(
        metadata.get("bundle_manifest_canonical_sha256"),
        f"{label}.manifest.canonical_sha256",
    )
    _require(
        raw_manifest == _EXPECTED_IDENTITIES["bundle_manifest_raw"]
        and canonical_manifest == _EXPECTED_IDENTITIES["bundle_manifest_canonical"],
        f"{label} bundle manifest does not match the pilot identity",
    )
    _require(
        bundle.get("role") == "parsed_bundle_manifest_trust_anchor"
        and bundle.get("external_expected_identity") is None
        and bundle.get("manifest_verified") is False,
        f"{label} overstates independent manifest verification",
    )
    parity = _mapping(manifest.get("parity_gate"), f"{label}.parity_gate")
    _require(
        parity.get("hard_gate") is True and parity.get("passed") is True,
        f"{label} export parity gate did not pass",
    )
    parameters = _integer(manifest.get("model_parameters"), f"{label}.parameters", 1)
    _require(metadata.get("model_parameters") == parameters, f"{label} parameters differ")
    bound["model_metadata"]["canonical_sha256"] = _sha256(
        metadata.get("model_meta_canonical_sha256"),
        f"{label}.model_meta_canonical_sha256",
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
                manifest.get("model_config_sha256"),
                f"{label}.model_config_sha256",
            ),
            "parameters": parameters,
        },
        "graph": {"file": graph_file, "bytes": graph_bytes, "sha256": graph},
        "bundle_manifest": {
            "raw_bytes": _integer(bundle.get("bytes"), f"{label}.manifest.bytes", 1),
            "raw_sha256": raw_manifest,
            "canonical_sha256": canonical_manifest,
            "schema_version": manifest["schema_version"],
        },
        **bound,
        "held_out_suite": {
            "file": suite_evidence["file"],
            "bytes": suite_bytes,
            "sha256": suite,
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


def _legacy_normalized_action(value: object, label: str) -> Mapping[str, Any]:
    action = _mapping(value, label)
    if action.get("abstain") is True:
        return {"abstain": True}
    return {"tool": action.get("tool"), "args": action.get("args")}


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
    *,
    label: str,
    benchmark_version: str,
) -> Mapping[str, Any]:
    result = _mapping(stored, label)
    errors = _list(result.get("errors"), f"{label}.errors")
    _require(all(isinstance(error, str) for error in errors), f"{label}.errors must be strings")
    future = benchmark_version == "rtab-dom-0.4"
    if future:
        _require(result.get("validator") in _DOM_VALIDATORS, f"{label}.validator differs")

    if action.get("abstain") is True:
        recomputed_errors = ["Abstention has no executable tool schema for this task suite."]
        schema_tool = None
        tool_schema = None
    else:
        schema_tool = result.get("schema_tool") if future else action.get("tool")
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
            if future:
                _require(
                    isinstance(schema_tool, str) and schema_tool == action.get("tool"),
                    f"{label}.schema_tool does not bind the predicted tool",
                )
            recomputed_errors = []
            _value_schema_errors(action.get("args"), tool_schema, "$.args", recomputed_errors)

    if future:
        expected = {
            "validator": result["validator"],
            "valid": len(recomputed_errors) == 0,
            "errors": recomputed_errors,
            "schema_tool": schema_tool,
            "tool_schema": tool_schema,
        }
    else:
        expected = {
            "valid": len(recomputed_errors) == 0,
            "errors": recomputed_errors,
            "tool_schema": tool_schema,
        }
    _require(dict(result) == expected, f"{label} does not reproduce from raw action and schema")
    return result


def _action_scores(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, bool]:
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


def _assertion_values(state: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    assertions = _list(state.get("assertions"), f"{label}.assertions")
    output = []
    for index, value in enumerate(assertions):
        assertion = _mapping(value, f"{label}.assertions[{index}]")
        _require(
            isinstance(assertion.get("passed"), bool),
            f"{label}.assertions[{index}].passed must be boolean",
        )
        output.append(
            {
                "document": assertion.get("document"),
                "selector": assertion.get("selector"),
                "kind": assertion.get("kind"),
                "actual": assertion.get("actual"),
            }
        )
    _require(
        state.get("passed") is all(assertion["passed"] is True for assertion in assertions),
        f"{label}.passed does not reproduce from assertions",
    )
    return output


def _validate_record(
    value: object,
    *,
    label: str,
    measured: bool,
    expected_repetition: int,
    expected_order: int,
    benchmark_version: str,
) -> Mapping[str, Any]:
    record = _mapping(value, label)
    _require(
        record.get("measured") is measured
        and record.get("repetition") == expected_repetition
        and record.get("order_index") == expected_order,
        f"{label} measurement order differs",
    )
    _require(
        record.get("backend") == "webgpu"
        and record.get("input_tokens") == INPUT_TOKENS,
        f"{label} execution protocol differs",
    )
    natural = _integer(record.get("natural_input_tokens"), f"{label}.natural_input_tokens")
    padding = _integer(record.get("context_padding_tokens"), f"{label}.padding_tokens")
    _require(natural + padding == INPUT_TOKENS, f"{label} padding accounting differs")
    if benchmark_version == "rtab-dom-0.4":
        _require(
            record.get("context_padding_placement") == "after_natural_assistant_marker"
            and record.get("decision_input_tokens") == natural
            and record.get("decision_feature_index") == natural - 1,
            f"{label} corrected decision-feature contract differs",
        )

    if benchmark_version == "rtab-dom-0.4":
        action = _normalized_action(record.get("predicted_action"), f"{label}.predicted_action")
        expected = _normalized_action(
            record.get("expected_action"), f"{label}.expected_action"
        )
        _require(
            record.get("expected") == expected,
            f"{label}.expected alias disagrees with expected_action",
        )
        parse = _mapping(record.get("parse_evidence"), f"{label}.parse_evidence")
        _require(
            parse.get("policy") == "structured_one_forward"
            and parse.get("inference_passes") == 1
            and parse.get("parse_kind") == "structured_heads",
            f"{label}.parse_evidence is not one structured inference pass",
        )
        for field in ("parse_failure", "runtime_validation_failure"):
            _require(isinstance(parse.get(field), bool), f"{label}.parse_evidence.{field} differs")
        _require(
            record.get("parse_failure") is parse["parse_failure"],
            f"{label}.parse_failure disagrees with parse evidence",
        )
        _require(
            record.get("action_timeout_ms") == 10_000
            and record.get("watchdog_outcome") == "completed_before_timeout",
            f"{label} watchdog evidence differs",
        )
    else:
        raw_action = _mapping(record.get("raw_model_output"), f"{label}.raw_model_output")
        _require(
            raw_action.get("policy") == "structured_one_forward"
            and raw_action.get("inference_passes") == 1
            and raw_action.get("parse_kind") == "structured_heads",
            f"{label} is not one structured inference pass",
        )
        action = _legacy_normalized_action(raw_action, f"{label}.raw_model_output")
        expected = _mapping(record.get("expected"), f"{label}.expected")
    action_scores = _action_scores(action, expected)
    independent_schema = _mapping(
        record.get("independent_schema"), f"{label}.independent_schema"
    )
    schema_result = _schema_result(
        action,
        independent_schema,
        label=f"{label}.independent_schema",
        benchmark_version=benchmark_version,
    )
    execution = _mapping(record.get("execution"), f"{label}.execution")
    before = _mapping(record.get("dom_before"), f"{label}.dom_before")
    after = _mapping(record.get("dom_after"), f"{label}.dom_after")
    before_values = _assertion_values(before, f"{label}.dom_before")
    after_values = _assertion_values(after, f"{label}.dom_after")
    fixture_clean = all(
        _mapping(value, f"{label}.dom_before.assertion").get("passed") is False
        for value in _list(before.get("assertions"), f"{label}.dom_before.assertions")
    )
    state_transition = (
        fixture_clean and before_values != after_values and after.get("passed") is True
    )
    score = _mapping(record.get("score"), f"{label}.score")
    expected_scores = {
        **action_scores,
        "schema_valid": schema_result.get("valid") is True,
        "fixture_clean": fixture_clean,
        "execution_ok": execution.get("ok") is True,
        "final_dom_valid": after.get("passed") is True,
        "state_transition": state_transition,
    }
    expected_scores["closed_loop_success"] = (
        expected_scores["exact_action"]
        and expected_scores["schema_valid"]
        and expected_scores["execution_ok"]
        and expected_scores["state_transition"]
    )
    for field, expected_value in expected_scores.items():
        _require(score.get(field) is expected_value, f"{label}.score.{field} differs")
    app_schema_valid = score.get("app_schema_valid_diagnostic")
    _require(
        isinstance(app_schema_valid, bool)
        and score.get("schema_validator_agreement")
        is (app_schema_valid is expected_scores["schema_valid"]),
        f"{label}.score schema-validator diagnostics differ",
    )
    _require(
        record.get("success") is expected_scores["exact_action"]
        and record.get("schema_valid") is expected_scores["schema_valid"],
        f"{label} root score aliases differ",
    )
    if benchmark_version == "rtab-dom-0.4":
        _require(
            record.get("validation_failure") is (not expected_scores["schema_valid"]),
            f"{label}.validation_failure disagrees with independent schema evidence",
        )
    predicted_tool = None if action.get("abstain") is True else action.get("tool")
    _require(
        record.get("predicted_tool") == predicted_tool
        and record.get("expected_tool") == expected.get("tool"),
        f"{label} tool aliases differ",
    )

    latency = _mapping(record.get("latency_ms"), f"{label}.latency_ms")
    for field in _LATENCY_FIELDS:
        _number(latency.get(field), f"{label}.latency_ms.{field}")
    aliases = {
        "harness_ttfa_ms": "harness_ttfa_ms",
        "runtime_ttfa_ms": "runtime_ttfa_ms",
        "independent_validate_ms": "independent_validate_ms",
    }
    for root_field, latency_field in aliases.items():
        _require(
            record.get(root_field) == latency.get(latency_field),
            f"{label}.{root_field} latency alias differs",
        )
    _require(
        record.get("ttfa_ms") == latency.get("harness_ttfa_ms"),
        f"{label}.ttfa_ms latency alias differs",
    )
    return record


def _validate_records(
    metadata: Mapping[str, Any],
    values: object,
    number: int,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    label = f"run {number}.records"
    raw = _list(values, label)
    _require(
        len(raw) == RECORDS_PER_RUN,
        f"{label} record count must be exactly {RECORDS_PER_RUN}",
    )
    records = [
        _validate_record(
            value,
            label=f"{label}[{index}]",
            measured=True,
            expected_repetition=index // CASES,
            expected_order=index % CASES,
            benchmark_version=str(metadata["benchmark_version"]),
        )
        for index, value in enumerate(raw)
    ]
    case_counts = Counter(str(record.get("case_id")) for record in records)
    action_counts = Counter(str(record.get("expected_tool")) for record in records)
    _require(
        len(case_counts) == CASES
        and set(case_counts.values()) == {REPETITIONS}
        and set(action_counts) == set(SUPPORTED_TOOLS)
        and set(action_counts.values()) == {REPETITIONS},
        f"{label} does not contain the exact case/action opportunity grid",
    )
    opportunities = Counter(
        (record["case_id"], record["repetition"]) for record in records
    )
    _require(
        len(opportunities) == RECORDS_PER_RUN and set(opportunities.values()) == {1},
        f"{label} opportunities are not unique and complete",
    )
    recorded_order = [
        {
            "repetition": record["repetition"],
            "order_index": record["order_index"],
            "case_id": record["case_id"],
        }
        for record in records
    ]
    _require(
        metadata.get("recorded_case_order") == recorded_order,
        f"{label} does not match metadata.recorded_case_order",
    )
    signatures = [
        canonical_sha256(
            {
                key: record[key]
                for key in (
                    "case_id",
                    "family",
                    "fixture",
                    "query",
                    "expected",
                    "repetition",
                    "order_index",
                    "input_bytes",
                    "natural_input_tokens",
                    "context_padding_tokens",
                )
            }
        )
        for record in records
    ]
    return records, signatures


def _validate_warmups(values: object, number: int, benchmark_version: str) -> None:
    label = f"run {number}.warmup_records"
    warmups = _list(values, label)
    _require(len(warmups) == WARMUPS, f"{label} count must be exactly {WARMUPS}")
    for index, phase in enumerate(("first_inference", "warmup", "warmup")):
        record = _validate_record(
            warmups[index],
            label=f"{label}[{index}]",
            measured=False,
            expected_repetition=-1,
            expected_order=index,
            benchmark_version=benchmark_version,
        )
        _require(record.get("phase") == phase, f"{label}[{index}] phase differs")


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
    for latency_name in ("harness_ttfa_ms", "closed_loop_ms"):
        latency_summary: dict[str, Any] = {}
        for percentile in ("p50", "p95", "p99"):
            values = [
                float(summary["latency_ms"][latency_name][percentile])
                for summary in summaries
            ]
            latency_summary[percentile] = {
                "by_run": values,
                "median": median(values),
                "range": [min(values), max(values)],
            }
        output[latency_name] = latency_summary
    return output


def _record_predicted_action(
    record: Mapping[str, Any],
    benchmark_version: str,
) -> Mapping[str, Any]:
    if benchmark_version == "rtab-dom-0.4":
        return _mapping(record.get("predicted_action"), "record.predicted_action")
    return _mapping(record.get("raw_model_output"), "record.raw_model_output")


def _outcome_evidence(benchmark_version: str) -> dict[str, str]:
    if benchmark_version == "rtab-dom-0.4":
        source = "normalized predicted_action and full expected_action"
        schema_source = "predicted_action and independent_schema.tool_schema"
    else:
        source = "historical raw_model_output and full expected action"
        schema_source = "historical raw_model_output and independent_schema.tool_schema"
    return {
        "mode": "independently_recomputed",
        "exact_action": (
            f"recomputed from {source}; stored exact-tool, exact-args, exact-action, and success "
            "booleans must agree"
        ),
        "schema_validity": (
            f"recomputed from {schema_source}; stored validator result, errors, schema-validity, "
            "and score aliases must agree"
        ),
        "historical_interpretation": (
            "rtab-dom-0.2 raw_model_output is used only as its original structured action object; "
            "no generated-text or post-hoc model-output reinterpretation is performed"
        ),
    }


def build_webgpu_dom_summary(
    raw_paths: Sequence[str | Path],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate exactly three DOM pilot runs and build their deterministic aggregate."""

    _require(len(raw_paths) == RUNS, f"expected exactly {RUNS} independent DOM runs")
    loaded = [_load(path, Path(repository_root)) for path in raw_paths]
    _require(
        len({run.artifact["sha256"] for run in loaded}) == RUNS,
        "raw DOM runs must have distinct file identities",
    )
    metadata: list[Mapping[str, Any]] = []
    identities: list[dict[str, Any]] = []
    records: list[list[Mapping[str, Any]]] = []
    signatures: list[list[str]] = []
    summaries: list[dict[str, Any]] = []
    for number, run in enumerate(loaded, start=1):
        run_metadata = _validate_protocol(run, number)
        identity = _validate_identity(run_metadata, number)
        _validate_warmups(
            run.payload.get("warmup_records"),
            number,
            str(run_metadata["benchmark_version"]),
        )
        run_records, signature = _validate_records(
            run_metadata,
            run.payload.get("records"),
            number,
        )
        summary = _summarize(run_records)
        _require(
            run.payload.get("summary") == summary,
            f"run {number} embedded summary does not reproduce from records",
        )
        metadata.append(run_metadata)
        identities.append(identity)
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
            f"run {index + 1} measured opportunity order differs from run 1",
        )
    created_at = [str(run.payload["created_at"]) for run in loaded]
    _require(
        created_at == sorted(created_at) and len(set(created_at)) == RUNS,
        "raw DOM runs must be supplied in distinct chronological order",
    )

    run_rows = []
    for number, (run, summary, run_records) in enumerate(
        zip(loaded, summaries, records, strict=True),
        start=1,
    ):
        run_rows.append(
            {
                "run": number,
                "created_at": run.payload["created_at"],
                "raw_artifact": copy.deepcopy(run.artifact),
                "records": len(run_records),
                "abstentions": sum(
                    _record_predicted_action(
                        record, str(metadata[number - 1]["benchmark_version"])
                    ).get("abstain")
                    is True
                    for record in run_records
                ),
                "ttfa_ms": copy.deepcopy(summary["latency_ms"]["harness_ttfa_ms"]),
                "closed_loop_ms": copy.deepcopy(summary["latency_ms"]["closed_loop_ms"]),
                "rates": {
                    field: summary[field]
                    for field in (
                        "exact_tool_rate",
                        "exact_args_rate",
                        "exact_action_rate",
                        "schema_valid_rate",
                        "final_dom_rate",
                        "state_transition_rate",
                        "closed_loop_success_rate",
                    )
                },
                "success_at_deadline_ms": copy.deepcopy(
                    summary["deadline_attainment_ms"]
                ),
            }
        )

    first = metadata[0]
    pooled_records = [record for run_records in records for record in run_records]
    pooled = _summarize(pooled_records)
    pooled_rates = {
        field: pooled[field]
        for field in (
            "exact_tool_rate",
            "exact_args_rate",
            "exact_action_rate",
            "schema_valid_rate",
            "final_dom_rate",
            "state_transition_rate",
            "closed_loop_success_rate",
        )
    }
    abstentions = sum(
        _record_predicted_action(record, str(first["benchmark_version"])).get("abstain") is True
        for record in pooled_records
    )
    report: dict[str, Any] = {
        "kind": WEBGPU_DOM_SUMMARY_KIND,
        "schema_version": WEBGPU_DOM_SUMMARY_SCHEMA_VERSION,
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
            "policy": "structured_one_forward",
            "inference_passes_per_record": 1,
            "precision": first["precision"],
            "target_input_tokens": first["target_input_tokens"],
            "warmups_per_run": first["warmups"],
            "cases": first["cases"],
            "repetitions_per_case": first["repetitions"],
            "records_per_run": first["measured_records"],
            "case_order_seed": first["case_order_seed"],
            "fixture_contract": copy.deepcopy(first["fixture_contract"]),
            "dispatch_contract": copy.deepcopy(first["dispatch_contract"]),
            "paint_barrier": first["paint_barrier"],
            "latency_clock": first["latency_clock"],
            "latency_boundaries": copy.deepcopy(first["latency_boundaries"]),
            "deadlines_ms": list(DEADLINES_MS),
            "success_at_deadline_definition": (
                "score.exact_action && score.schema_valid && harness_ttfa_ms <= deadline"
            ),
            "success_at_deadline_includes_dom_success": False,
            "dom_success_metric": "score.closed_loop_success",
            "record_outcome_evidence": _outcome_evidence(first["benchmark_version"]),
        },
        "identity": copy.deepcopy(identities[0]),
        "validation": {
            "status": "mechanically_valid",
            "checks": {
                "three_distinct_chronological_runs": True,
                "exact_pilot_artifact_identities": True,
                "webgpu_only_session_protocol": True,
                "exact_case_repetition_accounting": True,
                "fixed_512_token_context": True,
                "three_warmups_per_run": True,
                "one_structured_inference_pass_per_record": True,
                "finite_non_negative_ttfa_and_closed_loop_latency": True,
                "record_scores_reproduced": True,
                "embedded_summaries_reproduced": True,
                "record_outcome_evidence_scope_labeled": True,
            },
        },
        "runs": run_rows,
        "aggregate": {
            "run_count": RUNS,
            "records": len(pooled_records),
            "abstentions": abstentions,
            "all_predictions_abstained": abstentions == len(pooled_records),
            "score_counts": {
                field: sum(
                    _mapping(record["score"], "record.score").get(field) is True
                    for record in pooled_records
                )
                for field in _SCORE_FIELDS
            },
            "pooled_rates": pooled_rates,
            "run_level_latency_ms": _run_level(summaries),
            "pooled_measurement_unit": "measured single-step DOM opportunity",
            "pooled_metrics": pooled,
        },
    }
    if first["benchmark_version"] == "rtab-dom-0.4":
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


def write_webgpu_dom_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a finite, sorted summary after checking its self-hash."""

    payload = copy.deepcopy(dict(summary))
    recorded_hash = payload.pop("summary_sha256", None)
    _require(
        recorded_hash == canonical_sha256(payload),
        "WebGPU DOM summary self-hash is missing or invalid",
    )
    payload["summary_sha256"] = recorded_hash
    _finite_json(payload, "WebGPU DOM summary")
    encoded = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
