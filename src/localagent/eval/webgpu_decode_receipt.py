"""Fail-closed receipts for single-checkpoint WebGPU cached-decode acceptance runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from localagent.data.conversation_artifact import canonical_json_bytes


ACCEPTANCE_CONTEXT_LENGTHS = (128, 512, 1024, 1536)
ACCEPTANCE_OUTPUT_TOKENS = 32
ACCEPTANCE_WARMUPS = 3
ACCEPTANCE_REPETITIONS = 30
ACCEPTANCE_SEED = "slmw2026-cached-decode-v1"
ACCEPTANCE_SESSION_ORDER_SEED = f"{ACCEPTANCE_SEED}:single:session-create"
ACCEPTANCE_PROTOCOL_ID = "cached-decode-acceptance-1"
ACCEPTANCE_BENCHMARK = (
    "localagent_single_trained_cached_autoregressive_decode_acceptance_latency"
)
ACCEPTANCE_DECISION_ABI = "final_logits_argmax_with_next_token_crosscheck"
RECEIPT_KIND = "localagent_webgpu_cached_decode_acceptance_receipt"
RECEIPT_SCHEMA_VERSION = 3
HARNESS_SCHEMA_VERSION = 2
HARNESS_HTML_FILE = "decode-benchmark.html"
HARNESS_HTML_BYTES = 9_346
HARNESS_HTML_SHA256 = "390173bfef4ef310822cbefa591c9afa865a6348e3e69b4eb202b43addff09e5"
HARNESS_JAVASCRIPT_FILE = "decode-benchmark.js"
HARNESS_JAVASCRIPT_BYTES = 160_895
HARNESS_JAVASCRIPT_SHA256 = (
    "2cdbed45a26b2f7570a913f5bcbc7365e49eb64e442a9279682bf138685f6abe"
)
HARNESS_ORT_VERSION = "1.27.0"
HARNESS_ORT_VENDOR_PATH = "vendor/onnxruntime-web-1.27.0"
HARNESS_ORT_JAVASCRIPT_FILE = "ort.webgpu.min.js"
HARNESS_ORT_JAVASCRIPT_BYTES = 67_237
HARNESS_ORT_JAVASCRIPT_SHA256 = (
    "a3f348c2fec54c8c4ac503967c33c1943a79e96dba40fb867ab0f501be94bf84"
)
HARNESS_ORT_WASM_FILE = "ort-wasm-simd-threaded.jsep.wasm"
EVIDENCE_SCOPE = {
    "acquisition_bytes_externally_rooted": True,
    "browser_execution_attested": False,
    "gpu_hardware_attested": False,
    "scope": "externally_rooted_acquisition_bytes_with_unattested_browser_gpu_execution",
}
_SHA256_HEX_LENGTH = 64
MAX_WEBGPU_EVIDENCE_BYTES = 256 * 1024 * 1024
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)
_METRICS = (
    "ttft_ms",
    "tpot_ms",
    "decode_tokens_per_second",
    "prefill_ms",
    "decode_inference_ms",
    "model_decode_tokens_per_second",
)
_DTYPE_BYTES = {"float16": 2, "float32": 4, "int64": 8}
_RECEIPT_KEYS = {
    "benchmark_created_at",
    "bundle",
    "execution",
    "kind",
    "metrics_by_context",
    "model",
    "protocol",
    "receipt_self_sha256",
    "record_contract",
    "result",
    "runtime",
    "schema_version",
    "scope",
    "verified",
}
_DISPOSAL_COUNTER_KEYS = {
    "cache_tensors_allocated",
    "next_token_tensors_allocated",
    "logits_tensors_allocated",
    "decode_input_tensors_allocated",
    "cache_dispose_attempted",
    "cache_dispose_succeeded",
    "cache_dispose_failed",
    "cache_dispose_api_unavailable",
    "next_token_dispose_attempted",
    "next_token_dispose_succeeded",
    "next_token_dispose_failed",
    "next_token_dispose_api_unavailable",
    "logits_dispose_attempted",
    "logits_dispose_succeeded",
    "logits_dispose_failed",
    "logits_dispose_api_unavailable",
    "decode_input_dispose_attempted",
    "decode_input_dispose_succeeded",
    "decode_input_dispose_failed",
    "decode_input_dispose_api_unavailable",
    "superseded_cache_tensors_released",
    "final_cache_tensors_released",
}
_TRAINED_LABEL_KEYS = {
    "action_capability_claimed",
    "action_capability_evaluation",
    "artifact_manifest_latency_only",
    "benchmark_label",
    "capability_artifact",
    "latency_only",
    "quality_evaluation",
    "quality_scored_separately",
    "trained_weights",
    "untrained_random_weights",
}
_RAW_RESULT_KEYS = {
    *_TRAINED_LABEL_KEYS,
    "artifact_verification_records",
    "benchmark",
    "created_at",
    "errors",
    "failures",
    "input_preparation_record",
    "inputs",
    "metadata",
    "records",
    "schema_version",
    "session_records",
    "status",
    "summary",
    "warning",
    "warmup_records",
}
_METADATA_KEYS = {
    "acceptance_acquisition_roots",
    "acceptance_mode",
    "acceptance_protocol",
    "acceptance_wrapper_manifest_sha256",
    "action_capability_claimed",
    "action_capability_evaluated",
    "arm_count",
    "arms",
    "artifact_mode",
    "benchmark_label",
    "benchmark_mode",
    "benchmark_session_id",
    "browser",
    "cache_contract",
    "case_order_seed",
    "concurrency",
    "context_condition",
    "context_lengths",
    "cross_origin_isolated",
    "decision_output_abi",
    "decode_inference_ms_boundary",
    "device_memory_gb",
    "estimand",
    "evidence_scope",
    "excluded_from_latency",
    "external_machine_condition_sha256",
    "gpu",
    "graph_pass_contract",
    "greedy_selection",
    "hardware_concurrency",
    "harness_identity",
    "input_fixture_contract",
    "input_semantics",
    "language",
    "manifest",
    "manifest_raw_text",
    "manifest_sha256",
    "manifest_url",
    "measured_repetitions_per_condition",
    "model_decode_tokens_per_second_definition",
    "ort_script_url",
    "ort_version_pin",
    "ort_version_reported",
    "ort_version_verification_status",
    "ort_version_verified",
    "ort_wasm_num_threads",
    "ort_wasm_url",
    "output_tokens_per_condition",
    "page_to_ready_ms",
    "prefill_ms_boundary",
    "prompt_lengths_tokens",
    "protocol_version",
    "provider",
    "reported_percentiles",
    "required_webgpu_provider_verification",
    "run_challenge",
    "run_id",
    "run_once_reload_required",
    "session_order_seed",
    "shared_array_buffer_available",
    "tab_visibility_required",
    "timer",
    "tokenizer_asset",
    "tpot_boundary",
    "ttft_boundary",
    "user_agent",
    "verified_identities",
    "warmups_excluded_from_summary",
    "warmups_per_condition",
}
_SESSION_KEYS = {
    *_TRAINED_LABEL_KEYS,
    "arm_id",
    "benchmark_session_id",
    "cache_residency_requested",
    "error",
    "exact_provider_request_and_session_creation_observed",
    "execution_provider_list",
    "graph_bytes",
    "graph_kind",
    "graph_sha256",
    "graph_wide_provider_verified",
    "input_names",
    "logits_residency_requested",
    "next_token_residency_requested",
    "order_index",
    "ort_webgpu",
    "output_names",
    "per_node_fallback_status",
    "per_node_placement_status",
    "per_node_placement_verified",
    "phase",
    "preferred_output_location",
    "provider_actual",
    "provider_actual_observation",
    "provider_actual_scope",
    "provider_requested",
    "run_challenge",
    "session_create_ms",
    "whole_session_provider_retry",
}
_RECORD_KEYS = {
    *_TRAINED_LABEL_KEYS,
    "actual_graph_input_token_positions",
    "actual_input_tokens",
    "actual_output_tokens",
    "allocation_disposal",
    "arm_id",
    "benchmark_session_id",
    "cache",
    "decision_output_abi",
    "decode_inference_ms",
    "decode_pass_records",
    "decode_tokens_per_second",
    "decode_wall_ms",
    "disposal_contract_verified",
    "error",
    "generated_token_ids",
    "generated_token_interpretation",
    "generation_wall_ms",
    "global_order_index",
    "graph_bytes",
    "graph_files",
    "graph_pass_counts",
    "graph_sha256",
    "graph_wide_provider_verified",
    "input_tokens",
    "model_decode_tokens_per_second",
    "order_index",
    "output_tokens_requested",
    "pair_role",
    "per_node_fallback_status",
    "per_node_placement_verified",
    "phase",
    "prefill_ms",
    "prompt_tokens_actual",
    "prompt_tokens_requested",
    "provider_actual",
    "provider_actual_observation",
    "provider_requested",
    "repetition",
    "run_challenge",
    "run_id",
    "run_ok",
    "tpot_ms",
    "ttft_ms",
    "whole_session_provider_retry",
}
_GRAPH_PASS_KEYS = {
    "decode",
    "decode_attempted",
    "expected_decode",
    "expected_prefill",
    "expected_total",
    "prefill",
    "prefill_attempted",
    "total",
    "total_attempted",
}
_DECODE_PASS_KEYS = {
    "attention_cache_sequence_length",
    "cache_bound_directly_without_readback",
    "cache_logical_bytes_after",
    "cache_logical_bytes_before",
    "cache_reported_locations",
    "cache_tensor_count",
    "cache_tensors",
    "inference_ms",
    "input_token_id",
    "input_tokens",
    "output_token_id",
    "output_tokens",
    "pass_index",
    "pass_resolved_offset_ms",
    "pass_started_offset_ms",
    "token_available_ms",
    "token_available_offset_ms",
}
_CACHE_KEYS = {
    "cache_data_read_to_javascript",
    "dtype",
    "enabled",
    "final_logical_bytes",
    "final_tensors",
    "logits_residency",
    "next_token_residency",
    "next_token_role",
    "prefill_logical_bytes",
    "prefill_tensors",
    "requested_residency",
    "slot_count",
    "slots",
    "tensor_count",
    "token_selection_source",
    "update_strategy",
}
_CACHE_TENSOR_KEYS = {
    "dims",
    "dtype",
    "logical_bytes",
    "name",
    "reported_location",
}
_INPUT_KEYS = {
    "actual_tensor_tokens",
    "fixture_contract",
    "input_ids_int64_sha256",
    "input_semantics",
    "input_tokens",
    "tensor_dims",
    "tensor_dtype",
    "token_ids",
    "tokenizer_asset",
    "vocab_size",
}
_INPUT_PREPARATION_KEYS = {
    *_TRAINED_LABEL_KEYS,
    "all_actual_lengths_verified",
    "duration_ms",
    "fixture_contract",
    "input_semantics",
    "phase",
    "requested_context_lengths",
    "tokenizer_asset",
    "vocab_size",
}
_SUMMARY_KEYS = {
    "attempted",
    "completed",
    "conditions",
    "estimand",
    "failed",
    "quality_metrics_included",
}
_ORT_WEBGPU_KEYS = {
    "adapter_info",
    "ort_adapter_available",
    "ort_device_available",
}
_ADAPTER_INFO_KEYS = {
    "architecture",
    "description",
    "device",
    "is_fallback_adapter",
    "vendor",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{label} fields differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )


def _positive_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    number = float(value)
    _require(number > 0 if positive else number >= 0, f"{label} is out of range")
    return number


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _external_sha256(value: Any, label: str) -> str:
    digest = _sha256(value, label)
    _require(digest != "0" * _SHA256_HEX_LENGTH, f"{label} must not be the zero digest")
    return digest


def _uuid4(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _UUID4.fullmatch(value) is not None,
        f"{label} must be a lowercase UUIDv4",
    )
    return value


def _utc_timestamp(value: Any, label: str) -> datetime:
    _require(
        isinstance(value, str) and _UTC_TIMESTAMP.fullmatch(value) is not None,
        f"{label} must be an ISO-8601 UTC timestamp with millisecond precision",
    )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not a valid calendar timestamp") from error


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _read_stable_descriptor(
    descriptor: int,
    *,
    label: str,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
    _require(before.st_size <= max_bytes, f"{label} exceeds the input-size limit")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        _require(total <= max_bytes, f"{label} exceeds the input-size limit")
    after = os.fstat(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    _require(
        all(getattr(before, field) == getattr(after, field) for field in identity_fields),
        f"{label} changed while it was being read",
    )
    payload = b"".join(chunks)
    _require(len(payload) == after.st_size, f"{label} size changed while it was being read")
    return payload, after


def read_stable_webgpu_evidence_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int = MAX_WEBGPU_EVIDENCE_BYTES,
) -> bytes:
    """Read a stable regular file twice through no-follow descriptors."""

    _positive_integer(max_bytes, "maximum WebGPU evidence bytes")
    target = os.fspath(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    snapshots: list[tuple[bytes, os.stat_result]] = []
    for _ in range(2):
        try:
            descriptor = os.open(target, flags)
        except OSError as error:
            raise ValueError(f"{label} cannot be opened as a no-follow regular file") from error
        try:
            snapshots.append(
                _read_stable_descriptor(
                    descriptor,
                    label=label,
                    max_bytes=max_bytes,
                )
            )
        finally:
            os.close(descriptor)
    first_payload, first_stat = snapshots[0]
    second_payload, second_stat = snapshots[1]
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    _require(
        first_payload == second_payload
        and all(
            getattr(first_stat, field) == getattr(second_stat, field)
            for field in identity_fields
        ),
        f"{label} was not stable across independent reads",
    )
    return first_payload


def _isclose(actual: float, expected: float, label: str) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
        f"{label} does not match recomputed value",
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot summarize an empty metric")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _seeded_random_values(seed_text: str) -> Any:
    """Yield the exact uint32 Mulberry32 stream used by decode-benchmark.js."""

    state = 2_166_136_261
    for character in seed_text:
        state ^= ord(character)
        state = (state * 16_777_619) & 0xFFFF_FFFF
    while True:
        state = (state + 0x6D2B79F5) & 0xFFFF_FFFF
        value = state
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFF_FFFF
        mixed = ((value ^ (value >> 7)) * (value | 61)) & 0xFFFF_FFFF
        value = (value ^ ((value + mixed) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
        yield ((value ^ (value >> 14)) & 0xFFFF_FFFF) / 4_294_967_296


def _shuffled(values: Sequence[Any], seed_text: str) -> list[Any]:
    result = list(values)
    random_values = _seeded_random_values(seed_text)
    for index in range(len(result) - 1, 0, -1):
        replacement = math.floor(next(random_values) * (index + 1))
        result[index], result[replacement] = result[replacement], result[index]
    return result


def _expected_record_schedule(
    *,
    phase: str,
    repetitions: int,
    arm_id: Any,
) -> list[tuple[int, int, Any, int]]:
    output: list[tuple[int, int, Any, int]] = []
    conditions = [(arm_id, context) for context in ACCEPTANCE_CONTEXT_LENGTHS]
    for repetition in range(repetitions):
        order = _shuffled(conditions, f"{ACCEPTANCE_SEED}:{phase}:{repetition}")
        output.extend(
            (repetition, order_index, condition_arm, context)
            for order_index, (condition_arm, context) in enumerate(order)
        )
    return output


def _metric_summaries(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for context in ACCEPTANCE_CONTEXT_LENGTHS:
        matching = [record for record in records if record["input_tokens"] == context]
        metrics: dict[str, Any] = {}
        for metric in _METRICS:
            values = [
                _finite_number(record[metric], f"records.{metric}", positive=True)
                for record in matching
            ]
            metrics[metric] = {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
        cache_values = [
            _finite_number(
                _object(record["cache"], "record.cache")["final_logical_bytes"],
                "record.cache.final_logical_bytes",
                positive=True,
            )
            for record in matching
        ]
        metrics["final_logical_cache_bytes"] = {
            "p50": _percentile(cache_values, 0.50),
            "p95": _percentile(cache_values, 0.95),
        }
        output.append({"input_tokens": context, "metrics": metrics})
    return output


def _validate_no_capability_claims(
    value: Mapping[str, Any],
    label: str,
    *,
    require_artifact: bool = False,
) -> None:
    if require_artifact:
        _require(
            "capability_artifact" in value,
            f"{label}.capability_artifact must be explicitly false",
        )
    for field in (
        "capability_artifact",
        "action_capability_artifact",
        "action_capability_claimed",
        "action_capability_evaluated",
        "action_capability_evaluation",
    ):
        if field in value:
            _require(value[field] is False, f"{label}.{field} must be exactly false")
    for field in ("capability_claims", "action_capability_claims", "quality_claims"):
        if field in value:
            _require(value[field] == [], f"{label}.{field} must be exactly an empty array")
    for field in ("capability_metrics", "capability_artifact_type"):
        if field in value:
            _require(value[field] is None, f"{label}.{field} must be exactly null")


def _validate_manifest(
    metadata: Mapping[str, Any],
    *,
    expected_wrapper_manifest_sha256: str,
) -> tuple[Mapping[str, Any], str, int]:
    raw_text = metadata.get("manifest_raw_text")
    _require(isinstance(raw_text, str), "metadata.manifest_raw_text must be text")
    raw_payload = raw_text.encode("utf-8")
    digest = hashlib.sha256(raw_payload).hexdigest()
    expected_root = _external_sha256(
        expected_wrapper_manifest_sha256,
        "externally supplied wrapper manifest SHA-256",
    )
    _require(digest == metadata.get("manifest_sha256"), "manifest raw-text SHA-256 mismatch")
    _require(
        digest == metadata.get("acceptance_wrapper_manifest_sha256"),
        "browser result is not bound to its acceptance wrapper root",
    )
    _require(digest == expected_root, "result manifest differs from the external wrapper root")
    manifest = _object(
        _strict_json_bytes(raw_payload, label="embedded manifest"),
        "embedded manifest",
    )
    _require(manifest == metadata.get("manifest"), "parsed manifest disagrees with raw text")
    _require(
        _positive_integer(manifest.get("schema_version"), "manifest schema version") == 1
        and manifest.get("artifact_type") == "single_trained_cached_decode_suite"
        and manifest.get("trained") is True
        and manifest.get("latency_only") is False,
        "manifest is not a single trained cached-decode suite",
    )
    _validate_no_capability_claims(manifest, "manifest", require_artifact=True)
    quality = _object(manifest.get("quality_evaluation"), "manifest.quality_evaluation")
    _require(
        quality.get("included") is False and quality.get("required_separately") is True,
        "manifest must require separate quality evaluation",
    )
    model = _object(manifest.get("model"), "manifest.model")
    provenance_file = model.get("provenance")
    _require(
        isinstance(provenance_file, str) and provenance_file,
        "manifest.model.provenance must be non-empty",
    )
    artifacts = _object(manifest.get("artifacts"), "manifest.artifacts")
    _require(list(artifacts) == [provenance_file], "manifest must pin exactly one provenance file")
    pin = _object(artifacts[provenance_file], "manifest provenance pin")
    _exact_keys(pin, {"bytes", "sha256"}, "manifest provenance pin")
    _positive_integer(pin.get("bytes"), "manifest provenance bytes")
    _sha256(pin.get("sha256"), "manifest provenance SHA-256")
    return manifest, digest, len(raw_payload)


def _validate_provider(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    provider = _object(metadata.get("provider"), "metadata.provider")
    _exact_keys(
        provider,
        {
            "cache_output_location_verification_required",
            "exact_provider_request_and_session_creation_observed",
            "execution_provider_list",
            "graph_wide_provider_verified",
            "ort_webgpu",
            "per_node_fallback_status",
            "per_node_placement_status",
            "per_node_placement_verified",
            "provider_actual",
            "provider_actual_observation",
            "provider_actual_scope",
            "provider_requested",
            "required_for_single_model",
            "required_verification_passed",
            "verification_method",
            "whole_session_provider_retry",
        },
        "metadata.provider",
    )
    _require(
        provider.get("provider_requested") == "webgpu"
        and provider.get("provider_actual") is None
        and provider.get("execution_provider_list") == ["webgpu"]
        and provider.get("whole_session_provider_retry") is False
        and provider.get("exact_provider_request_and_session_creation_observed") is True
        and provider.get("graph_wide_provider_verified") is False
        and provider.get("per_node_placement_verified") is False
        and provider.get("per_node_placement_status") == "unknown"
        and provider.get("per_node_fallback_status") == "unknown"
        and provider.get("required_for_single_model") is True
        and provider.get("required_verification_passed") is True,
        "result lacks exact WebGPU request/session evidence with unknown placement",
    )
    observation = provider.get("provider_actual_observation")
    scope = provider.get("provider_actual_scope")
    _require(
        isinstance(observation, str)
        and "not exposed" in observation
        and isinstance(scope, str)
        and "unknown" in scope,
        "provider evidence overstates actual execution placement",
    )
    ort_webgpu = _validate_ort_webgpu_evidence(
        provider.get("ort_webgpu"),
        "metadata.provider.ort_webgpu",
    )
    _require(
        ort_webgpu.get("ort_adapter_available") is True
        and ort_webgpu.get("ort_device_available") is True,
        "ORT WebGPU adapter/device were not observed",
    )
    _require(
        metadata.get("required_webgpu_provider_verification") is True,
        "single acceptance run did not require WebGPU verification",
    )
    gpu = _object(metadata.get("gpu"), "metadata.gpu")
    _require(
        gpu.get("navigator_gpu_available") is True
        and gpu.get("ort_webgpu") == ort_webgpu,
        "runtime GPU metadata disagrees with provider evidence",
    )
    version_pin = metadata.get("ort_version_pin")
    version_reported = metadata.get("ort_version_reported")
    _require(
        isinstance(version_pin, str)
        and bool(version_pin)
        and isinstance(version_reported, str)
        and bool(version_reported)
        and metadata.get("ort_version_verified") is True
        and version_pin == version_reported,
        "ONNX Runtime Web version pin was not positively verified",
    )
    return provider


def _validate_ort_webgpu_evidence(value: Any, label: str) -> Mapping[str, Any]:
    evidence = _object(value, label)
    _exact_keys(evidence, _ORT_WEBGPU_KEYS, label)
    adapter_info = _object(evidence.get("adapter_info"), f"{label}.adapter_info")
    _exact_keys(adapter_info, _ADAPTER_INFO_KEYS, f"{label}.adapter_info")
    for field in ("architecture", "description", "device", "vendor"):
        observed = adapter_info.get(field)
        _require(
            observed is None or isinstance(observed, str),
            f"{label}.adapter_info.{field} must be a string or null",
        )
    fallback = adapter_info.get("is_fallback_adapter")
    _require(
        fallback is None or isinstance(fallback, bool),
        f"{label}.adapter_info.is_fallback_adapter must be a boolean or null",
    )
    return evidence


def _validate_harness_resource(
    value: Any,
    *,
    label: str,
    expected_file: str,
    expected_sha256: str,
    expected_bytes: int | None = None,
    expected_url_suffix: str | None = None,
) -> dict[str, Any]:
    resource = _object(value, label)
    _exact_keys(
        resource,
        {
            "bytes",
            "external_expected_sha256",
            "hash_verified",
            "relative_path",
            "sha256",
            "url",
        },
        label,
    )
    _require(resource.get("relative_path") == expected_file, f"{label} file identity differs")
    observed_bytes = _positive_integer(resource.get("bytes"), f"{label}.bytes")
    if expected_bytes is not None:
        _require(
            observed_bytes == expected_bytes,
            f"{label} byte count differs from the source-controlled harness",
        )
    _require(
        _sha256(resource.get("external_expected_sha256"), f"{label}.external_expected_sha256")
        == expected_sha256
        and _sha256(resource.get("sha256"), f"{label}.sha256") == expected_sha256
        and resource.get("hash_verified") is True,
        f"{label} SHA-256 differs from its external acquisition root",
    )
    url = resource.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    url_suffix = expected_url_suffix or f"/{expected_file}"
    _require(
        parsed is not None
        and parsed.scheme in {"http", "https"}
        and parsed.path.endswith(url_suffix)
        and not parsed.fragment,
        f"{label}.url is not an absolute HTTP(S) harness URL",
    )
    return dict(resource)


def _validate_harness_identity(
    metadata: Mapping[str, Any],
    *,
    expected_html_sha256: str,
    expected_javascript_sha256: str,
    expected_ort_javascript_sha256: str,
    expected_ort_wasm_sha256: str,
) -> dict[str, Any]:
    html_sha256 = _external_sha256(expected_html_sha256, "expected harness HTML SHA-256")
    javascript_sha256 = _external_sha256(
        expected_javascript_sha256,
        "expected harness JavaScript SHA-256",
    )
    ort_javascript_sha256 = _external_sha256(
        expected_ort_javascript_sha256,
        "expected ORT JavaScript SHA-256",
    )
    ort_wasm_sha256 = _external_sha256(
        expected_ort_wasm_sha256,
        "expected ORT WASM SHA-256",
    )
    _require(
        html_sha256 == HARNESS_HTML_SHA256
        and javascript_sha256 == HARNESS_JAVASCRIPT_SHA256,
        "external harness roots differ from the source-controlled HTML/JavaScript",
    )
    _require(
        ort_javascript_sha256 == HARNESS_ORT_JAVASCRIPT_SHA256,
        "external ORT JavaScript root differs from the SRI-pinned runtime bytes",
    )
    _require(
        metadata.get("acceptance_acquisition_roots")
        == {
            "html_sha256": html_sha256,
            "javascript_sha256": javascript_sha256,
            "ort_javascript_sha256": ort_javascript_sha256,
            "ort_wasm_sha256": ort_wasm_sha256,
        },
        "metadata acquisition-root declaration differs from external roots",
    )
    harness = _object(metadata.get("harness_identity"), "metadata.harness_identity")
    _exact_keys(
        harness,
        {"html", "javascript", "ort", "schema_version"},
        "metadata.harness_identity",
    )
    _require(
        harness.get("schema_version") == HARNESS_SCHEMA_VERSION,
        "harness identity schema is unsupported",
    )
    html = _validate_harness_resource(
        harness.get("html"),
        label="metadata.harness_identity.html",
        expected_file=HARNESS_HTML_FILE,
        expected_bytes=HARNESS_HTML_BYTES,
        expected_sha256=html_sha256,
    )
    javascript = _validate_harness_resource(
        harness.get("javascript"),
        label="metadata.harness_identity.javascript",
        expected_file=HARNESS_JAVASCRIPT_FILE,
        expected_bytes=HARNESS_JAVASCRIPT_BYTES,
        expected_sha256=javascript_sha256,
    )
    ort = _object(harness.get("ort"), "metadata.harness_identity.ort")
    _exact_keys(
        ort,
        {
            "javascript",
            "self_hosted_same_origin",
            "version_pin",
            "version_reported",
            "version_verified",
            "wasm",
        },
        "metadata.harness_identity.ort",
    )
    ort_javascript = _validate_harness_resource(
        ort.get("javascript"),
        label="metadata.harness_identity.ort.javascript",
        expected_file=HARNESS_ORT_JAVASCRIPT_FILE,
        expected_bytes=HARNESS_ORT_JAVASCRIPT_BYTES,
        expected_sha256=ort_javascript_sha256,
        expected_url_suffix=f"/{HARNESS_ORT_VENDOR_PATH}/{HARNESS_ORT_JAVASCRIPT_FILE}",
    )
    ort_wasm = _validate_harness_resource(
        ort.get("wasm"),
        label="metadata.harness_identity.ort.wasm",
        expected_file=HARNESS_ORT_WASM_FILE,
        expected_sha256=ort_wasm_sha256,
        expected_url_suffix=f"/{HARNESS_ORT_VENDOR_PATH}/{HARNESS_ORT_WASM_FILE}",
    )
    origins = {
        (urlparse(resource["url"]).scheme, urlparse(resource["url"]).netloc)
        for resource in (html, javascript, ort_javascript, ort_wasm)
    }
    _require(
        len(origins) == 1
        and ort.get("self_hosted_same_origin") is True
        and ort.get("version_pin") == HARNESS_ORT_VERSION
        and ort.get("version_reported") == HARNESS_ORT_VERSION
        and ort.get("version_verified") is True
        and metadata.get("ort_script_url") == ort_javascript["url"]
        and metadata.get("ort_wasm_url") == ort_wasm["url"]
        and metadata.get("ort_version_pin") == HARNESS_ORT_VERSION
        and metadata.get("ort_version_reported") == HARNESS_ORT_VERSION
        and metadata.get("ort_version_verified") is True,
        "ONNX Runtime acquisition bytes are not self-hosted, same-origin, and verified",
    )
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "html": html,
        "javascript": javascript,
        "ort": {
            "javascript": ort_javascript,
            "self_hosted_same_origin": True,
            "version_pin": HARNESS_ORT_VERSION,
            "version_reported": HARNESS_ORT_VERSION,
            "version_verified": True,
            "wasm": ort_wasm,
        },
    }


def _artifact_records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = [
        _object(value, f"artifact records[{index}]")
        for index, value in enumerate(
            _array(result.get("artifact_verification_records"), "artifact records")
        )
    ]
    _require(bool(records), "artifact verification records are empty")
    for record in records:
        _sha256(record.get("actual_sha256"), "artifact actual SHA-256")
        _positive_integer(record.get("bytes"), "artifact bytes")
        expected_sha256 = record.get("expected_sha256")
        if expected_sha256 is not None:
            _sha256(expected_sha256, "artifact expected SHA-256")
            _require(
                expected_sha256 == record.get("actual_sha256")
                and record.get("hash_verified") is True,
                "artifact SHA-256 pin was not verified",
            )
        expected_bytes = record.get("expected_bytes")
        if expected_bytes is not None:
            _positive_integer(expected_bytes, "artifact expected bytes")
            _require(
                expected_bytes == record.get("bytes")
                and record.get("bytes_verified") is True,
                "artifact byte-count pin was not verified",
            )
        _require(
            record.get("verification_before_parse_or_ort") is True,
            "artifact was not verified before parse/ORT use",
        )
    return records


def _find_artifact_record(
    records: Sequence[Mapping[str, Any]],
    *,
    artifact_kind: str,
    relative_path: str,
    sha256: str,
    size: int,
    externally_rooted: bool = False,
) -> Mapping[str, Any]:
    matches = [
        record
        for record in records
        if record.get("artifact_kind") == artifact_kind
        and record.get("relative_path") == relative_path
        and record.get("actual_sha256") == sha256
        and record.get("bytes") == size
    ]
    _require(
        len(matches) == 1,
        f"artifact records do not bind exactly one {artifact_kind} {relative_path}",
    )
    record = matches[0]
    _require(
        record.get("expected_sha256") == sha256 and record.get("hash_verified") is True,
        f"{artifact_kind} does not carry an externally or provenance-pinned SHA-256",
    )
    if externally_rooted:
        _require(
            record.get("hash_verification_status") == "verified_by_external_acceptance_root",
            "wrapper manifest was not verified against the external acceptance root",
        )
    else:
        _require(
            record.get("expected_bytes") == size and record.get("bytes_verified") is True,
            f"{artifact_kind} does not carry a verified byte-count pin",
        )
    return record


def _find_wrapper_record(
    records: Sequence[Mapping[str, Any]],
    *,
    sha256: str,
    size: int,
) -> Mapping[str, Any]:
    matches = [
        record
        for record in records
        if record.get("artifact_kind") == "single_decode_manifest"
        and record.get("actual_sha256") == sha256
        and record.get("bytes") == size
    ]
    _require(len(matches) == 1, "artifact records do not bind exactly one wrapper manifest")
    record = matches[0]
    _require(
        isinstance(record.get("relative_path"), str)
        and bool(record["relative_path"])
        and record.get("expected_sha256") == sha256
        and record.get("hash_verified") is True
        and record.get("hash_verification_status")
        == "verified_by_external_acceptance_root",
        "wrapper manifest was not verified against the external acceptance root",
    )
    return record


def _artifact_pin(
    provenance: Mapping[str, Any],
    filename: str,
    *,
    label: str,
) -> tuple[int, str]:
    artifacts = _object(provenance.get("artifacts"), "provenance.artifacts")
    pin = _object(artifacts.get(filename), label)
    _require(pin.get("file") == filename, f"{label}.file mismatch")
    return (
        _positive_integer(pin.get("bytes"), f"{label}.bytes"),
        _sha256(pin.get("sha256"), f"{label}.sha256"),
    )


def _validate_training_identity(value: Any, index: int) -> dict[str, Any]:
    identity = _object(value, f"training artifact identity {index}")
    _exact_keys(
        identity,
        {"artifact_kind", "bytes", "path", "sha256"},
        f"training artifact identity {index}",
    )
    artifact_kind = identity.get("artifact_kind")
    path = identity.get("path")
    _require(
        isinstance(artifact_kind, str) and bool(artifact_kind),
        f"training artifact identity {index} kind must be non-empty",
    )
    _require(
        isinstance(path, str) and path.startswith("/"),
        f"training artifact identity {index} path must be absolute",
    )
    return {
        "artifact_kind": artifact_kind,
        "bytes": _positive_integer(identity.get("bytes"), "training artifact bytes"),
        "path": path,
        "sha256": _sha256(identity.get("sha256"), "training artifact SHA-256"),
    }


def _validate_modern_graph_and_parity(
    provenance: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> None:
    model = _object(provenance.get("model"), "provenance.model")
    config = _object(model.get("config"), "provenance.model.config")
    _require(config == arm.get("config"), "arm config differs from raw provenance")
    vocab_size = _positive_integer(config.get("vocab_size"), "provenance vocabulary size")
    graph_contract = _object(provenance.get("graph_contract"), "provenance.graph_contract")
    next_token = _object(graph_contract.get("next_token"), "graph next_token contract")
    logits = _object(graph_contract.get("logits"), "graph logits contract")
    _require(
        next_token.get("name") == "next_token"
        and next_token.get("dtype") == "int64"
        and next_token.get("shape") == ["batch"]
        and next_token.get("decode")
        == "compatibility argmax over the exported final-token logits"
        and logits.get("name") == "logits"
        and logits.get("shape") == ["batch", vocab_size],
        "raw provenance does not declare the final-logits decision ABI",
    )
    slots = [
        _object(value, "cache slot")
        for value in _array(graph_contract.get("cache_slots"), "provenance cache slots")
    ]
    present_names = [
        name
        for slot in slots
        for name in _array(slot.get("present_outputs"), "cache present outputs")
    ]
    precision = arm.get("precision")
    graphs = _object(graph_contract.get("graphs"), "provenance graph precisions")
    precision_graph = _object(graphs.get(precision), f"provenance graph {precision}")
    expected_outputs = ["next_token", "logits", *present_names]
    for graph_kind in ("prefill", "decode"):
        graph = _object(precision_graph.get(graph_kind), f"{precision}.{graph_kind}")
        _require(
            graph.get("file") == arm.get(f"{graph_kind}_file")
            and graph.get("output_names") == expected_outputs,
            f"{graph_kind} graph I/O does not expose the accepted logits/cache ABI",
        )
    parity = _object(provenance.get("parity"), "provenance.parity")
    results = _object(parity.get("results"), "provenance.parity.results")
    parity_result = _object(results.get(precision), f"provenance parity {precision}")
    _require(
        parity.get("hard_gate") is True
        and parity_result.get("hard_gate") is True
        and parity_result.get("passed") is True
        and parity_result.get("greedy_next_token_exact") is True
        and parity_result.get("final_token_logits_shape") == ["batch", vocab_size]
        and parity_result.get("reference")
        == "exact in-memory LocalAgentLM checkpoint weights",
        "selected graph precision lacks hard checkpoint trajectory parity",
    )
    parity_artifacts = _object(
        parity_result.get("artifacts"),
        f"provenance parity {precision}.artifacts",
    )
    for graph_kind in ("prefill", "decode"):
        filename = arm.get(f"{graph_kind}_file")
        size, sha256 = _artifact_pin(
            provenance,
            str(filename),
            label=f"provenance.artifacts.{filename}",
        )
        _require(
            parity_artifacts.get(graph_kind) == {"bytes": size, "sha256": sha256},
            f"{graph_kind} graph identity is not bound to trajectory parity",
        )


def _validate_arm_provenance(
    arm: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    Mapping[str, Any],
    str,
    str,
    list[dict[str, Any]],
    dict[str, Any],
]:
    raw_text = arm.get("provenance_raw_text")
    _require(isinstance(raw_text, str), "arm provenance raw text is unavailable")
    raw_payload = raw_text.encode("utf-8")
    raw_sha256 = hashlib.sha256(raw_payload).hexdigest()
    provenance_sha256 = _sha256(arm.get("provenance_sha256"), "provenance SHA-256")
    provenance_bytes = _positive_integer(arm.get("provenance_bytes"), "provenance bytes")
    _require(
        raw_sha256 == provenance_sha256 and len(raw_payload) == provenance_bytes,
        "arm provenance raw bytes do not match the embedded identity",
    )
    provenance = _object(
        _strict_json_bytes(raw_payload, label="embedded provenance"),
        "embedded provenance",
    )
    _require(provenance == arm.get("provenance"), "parsed provenance disagrees with raw text")
    _require(
        _positive_integer(provenance.get("schema_version"), "provenance schema version") == 1
        and provenance.get("artifact_type") == "trained_checkpoint_cached_decode_onnx"
        and provenance.get("trained") is True
        and provenance.get("latency_only") is False,
        "provenance is not checkpoint-backed trained cached decode",
    )
    _validate_no_capability_claims(provenance, "provenance", require_artifact=True)
    model = _object(provenance.get("model"), "provenance.model")
    manifest_model = _object(manifest.get("model"), "manifest.model")
    _require(
        model.get("name") == arm.get("id") == manifest_model.get("name")
        and model.get("pair_role") == arm.get("pair_role") == manifest_model.get("pair_role"),
        "manifest, arm, and provenance model identities disagree",
    )
    provenance_file = manifest_model["provenance"]
    manifest_pin = _object(
        _object(manifest.get("artifacts"), "manifest.artifacts")[provenance_file],
        "manifest provenance pin",
    )
    _require(
        manifest_pin.get("sha256") == provenance_sha256
        and manifest_pin.get("bytes") == provenance_bytes
        and arm.get("provenance_file") == provenance_file,
        "raw provenance is not bound to the wrapper manifest pin",
    )
    _find_artifact_record(
        records,
        artifact_kind="model_provenance",
        relative_path=provenance_file,
        sha256=provenance_sha256,
        size=provenance_bytes,
    )

    _require(
        arm.get("decision_output_abi") == ACCEPTANCE_DECISION_ABI,
        "acceptance requires the final-logits decision ABI",
    )
    _validate_modern_graph_and_parity(provenance, arm)
    precision = arm.get("precision")
    _require(precision in {"fp32", "fp16"}, "arm precision is unsupported")
    graph_contract = _object(provenance.get("graph_contract"), "provenance.graph_contract")
    graphs = _object(graph_contract.get("graphs"), "provenance.graph_contract.graphs")
    precision_graph = _object(graphs.get(precision), f"provenance graph {precision}")
    graph_identities: dict[str, Any] = {}
    for graph_kind in ("prefill", "decode"):
        graph = _object(precision_graph.get(graph_kind), f"{precision}.{graph_kind}")
        filename = graph.get("file")
        _require(isinstance(filename, str) and bool(filename), f"{graph_kind} file is missing")
        size, sha256 = _artifact_pin(
            provenance,
            filename,
            label=f"provenance.artifacts.{filename}",
        )
        _require(
            arm.get(f"{graph_kind}_file") == filename
            and _positive_integer(
                arm.get(f"{graph_kind}_bytes"),
                f"arm {graph_kind} graph bytes",
            )
            == size
            and arm.get(f"{graph_kind}_sha256") == sha256,
            f"arm {graph_kind} graph identity disagrees with raw provenance",
        )
        _find_artifact_record(
            records,
            artifact_kind=f"cached_{graph_kind}_onnx_graph",
            relative_path=filename,
            sha256=sha256,
            size=size,
        )
        graph_identities[graph_kind] = {
            "file": filename,
            "bytes": size,
            "sha256": sha256,
        }

    weights = _object(provenance.get("weights"), "provenance.weights")
    checkpoint_sha256 = _sha256(weights.get("checkpoint_sha256"), "checkpoint SHA-256")
    _require(checkpoint_sha256 == arm.get("checkpoint_sha256"), "arm checkpoint pin mismatch")
    stage = weights.get("checkpoint_stage")
    _require(stage in {"pretrain", "midtrain", "sft", "rl"}, "checkpoint stage is unsupported")

    lineage = arm.get("training_lineage")
    lineage_raw_text = arm.get("training_lineage_raw_text")
    training_artifacts: list[dict[str, Any]] = []
    lineage_identity: dict[str, Any] = {
        "file": None,
        "bytes": None,
        "sha256": None,
    }
    if stage == "pretrain" and provenance.get("training_lineage_export") is None:
        _require(
            lineage is None and lineage_raw_text is None,
            "pretrain provenance without a lineage pin cannot embed lineage",
        )
    else:
        lineage_file = provenance.get("training_lineage_export")
        _require(
            lineage_file == "training-lineage.json",
            "checkpoint provenance has no canonical training-lineage pin",
        )
        _require(isinstance(lineage_raw_text, str), "training-lineage raw text is unavailable")
        lineage_payload = lineage_raw_text.encode("utf-8")
        lineage_bytes, lineage_sha256 = _artifact_pin(
            provenance,
            lineage_file,
            label="provenance.artifacts.training-lineage.json",
        )
        _require(
            len(lineage_payload) == lineage_bytes
            and hashlib.sha256(lineage_payload).hexdigest() == lineage_sha256,
            "training-lineage raw bytes do not match provenance pins",
        )
        parsed_lineage = _object(
            _strict_json_bytes(lineage_payload, label="embedded training lineage"),
            "embedded training lineage",
        )
        _require(parsed_lineage == lineage, "parsed training lineage disagrees with raw text")
        _require(
            parsed_lineage.get("kind") == "localagent_training_lineage_export"
            and _positive_integer(
                parsed_lineage.get("schema_version"),
                "training lineage schema version",
            )
            == 1
            and parsed_lineage.get("stage") == stage
            and parsed_lineage.get("checkpoint_sha256") == checkpoint_sha256,
            "training lineage is not bound to provenance stage/checkpoint",
        )
        lineage_core = _object(parsed_lineage.get("lineage"), "training lineage core")
        _require(
            lineage_core.get("stage") == stage,
            "training lineage core stage differs from checkpoint stage",
        )
        checkpoint_lineage = provenance.get("checkpoint_lineage")
        _require(
            checkpoint_lineage == lineage_core,
            "raw provenance checkpoint lineage differs from training-lineage bytes",
        )
        tokenizer = _object(provenance.get("tokenizer"), "provenance.tokenizer")
        _require(
            lineage_core.get("tokenizer_sha256") == tokenizer.get("sha256"),
            "training lineage tokenizer differs from provenance",
        )
        _find_artifact_record(
            records,
            artifact_kind="training_lineage",
            relative_path=lineage_file,
            sha256=lineage_sha256,
            size=lineage_bytes,
        )
        hashes = [
            _sha256(value, "training artifact SHA-256")
            for value in _array(
                parsed_lineage.get("training_artifact_sha256"),
                "training artifact hashes",
            )
        ]
        _require(len(set(hashes)) == len(hashes), "training artifact hashes are not unique")
        raw_identities = parsed_lineage.get("training_artifacts")
        if stage in {"midtrain", "sft", "rl"}:
            _require(bool(hashes), "accepted posttraining lineage has no training artifact hashes")
            identities = _array(raw_identities, "training artifact identities")
            _require(
                len(identities) == len(hashes),
                "training artifact identity/hash cardinality mismatch",
            )
            training_artifacts = [
                _validate_training_identity(value, index)
                for index, value in enumerate(identities)
            ]
            _require(
                [identity["sha256"] for identity in training_artifacts] == hashes,
                "training artifact identities do not match lineage hash order",
            )
            _require(
                len({identity["path"] for identity in training_artifacts})
                == len(training_artifacts),
                "training artifact identity paths are not unique",
            )
            _require(
                parsed_lineage.get("conversation_prompt_contract") == "openai_full_catalog_v1",
                "posttraining lineage prompt contract is unsupported",
            )
        else:
            _require(
                parsed_lineage.get("conversation_prompt_contract") is None,
                "pretrain lineage must have a null conversation prompt contract",
            )
            if hashes:
                identities = _array(raw_identities, "pretrain training artifact identities")
                training_artifacts = [
                    _validate_training_identity(value, index)
                    for index, value in enumerate(identities)
                ]
                _require(
                    [identity["sha256"] for identity in training_artifacts] == hashes,
                    "pretrain training artifact identities disagree with hashes",
                )
                _require(
                    len({identity["path"] for identity in training_artifacts})
                    == len(training_artifacts),
                    "pretrain training artifact identity paths are not unique",
                )
            else:
                _require(
                    raw_identities in (None, []),
                    "pretrain lineage has identities without training artifact hashes",
                )
        lineage_identity = {
            "file": lineage_file,
            "bytes": lineage_bytes,
            "sha256": lineage_sha256,
        }
    return (
        provenance,
        checkpoint_sha256,
        stage,
        training_artifacts,
        {
            "provenance": {
                "file": provenance_file,
                "bytes": provenance_bytes,
                "sha256": provenance_sha256,
            },
            "lineage": lineage_identity,
            **graph_identities,
        },
    )


def _validate_inputs(result: Mapping[str, Any], *, vocab_size: int) -> None:
    preparation = _object(result.get("input_preparation_record"), "input preparation record")
    _exact_keys(
        preparation,
        _INPUT_PREPARATION_KEYS,
        "input preparation record",
    )
    _require(
        preparation.get("requested_context_lengths") == list(ACCEPTANCE_CONTEXT_LENGTHS)
        and preparation.get("all_actual_lengths_verified") is True,
        "input preparation did not verify every acceptance context length",
    )
    inputs = _array(result.get("inputs"), "inputs")
    _require(len(inputs) == len(ACCEPTANCE_CONTEXT_LENGTHS), "input fixture count mismatch")
    by_context: dict[int, Mapping[str, Any]] = {}
    for value in inputs:
        fixture = _object(value, "input fixture")
        _exact_keys(fixture, _INPUT_KEYS, "input fixture")
        context = _positive_integer(fixture.get("input_tokens"), "input tokens")
        _require(context not in by_context, "input fixture contexts are duplicated")
        by_context[context] = fixture
    _require(tuple(sorted(by_context)) == ACCEPTANCE_CONTEXT_LENGTHS, "input contexts mismatch")
    for context, fixture in by_context.items():
        expected_ids = [(131 * index + 17) % vocab_size for index in range(context)]
        token_ids = _array(fixture.get("token_ids"), f"input IDs at context {context}")
        tensor_dims = _array(fixture.get("tensor_dims"), f"input tensor dims at context {context}")
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids)
            and token_ids == expected_ids,
            f"input IDs differ at context {context}",
        )
        _require(
            _positive_integer(
                fixture.get("actual_tensor_tokens"),
                f"input actual tensor tokens at context {context}",
            )
            == context
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in tensor_dims
            )
            and tensor_dims == [1, context]
            and fixture.get("tensor_dtype") == "int64",
            f"input tensor contract differs at context {context}",
        )
        encoded = b"".join(struct.pack("<q", value) for value in expected_ids)
        expected_sha256 = hashlib.sha256(encoded).hexdigest()
        _require(
            fixture.get("input_ids_int64_sha256") == expected_sha256,
            f"input tensor SHA-256 differs at context {context}",
        )


def _tensor_logical_bytes(
    values: Any,
    label: str,
    *,
    expected_names: Sequence[str],
    expected_dtype: str,
    expected_dims: Mapping[str, Sequence[int]],
) -> tuple[int, int]:
    tensors = _array(values, label)
    total = 0
    names: list[str] = []
    for index, value in enumerate(tensors):
        tensor = _object(value, f"{label}[{index}]")
        _exact_keys(tensor, _CACHE_TENSOR_KEYS, f"{label}[{index}]")
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        dims = _array(tensor.get("dims"), f"{label}[{index}].dims")
        _require(
            isinstance(name, str) and bool(name),
            f"{label}[{index}].name must be non-empty",
        )
        _require(dtype == expected_dtype, f"{label}[{index}].dtype is inconsistent")
        _require(
            bool(dims)
            and all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in dims
            ),
            f"{label}[{index}].dims must be positive integers",
        )
        _require(
            name in expected_dims and dims == list(expected_dims[name]),
            f"{label}[{index}].dims differ from the exact cache-slot shape",
        )
        computed = math.prod(dims) * _DTYPE_BYTES[expected_dtype]
        logical_bytes = _positive_integer(
            tensor.get("logical_bytes"),
            f"{label}[{index}].logical_bytes",
        )
        _require(
            logical_bytes == computed,
            f"{label}[{index}] logical bytes do not match dtype/dimensions",
        )
        _require(
            tensor.get("reported_location") == "gpu-buffer",
            f"{label}[{index}] was not observed at gpu-buffer",
        )
        names.append(name)
        total += logical_bytes
    _require(names == list(expected_names), f"{label} names/order differ from graph provenance")
    return len(tensors), total


def _cache_contract_identity(
    arm: Mapping[str, Any],
    *,
    attention_sequence: int,
) -> tuple[list[str], str, dict[str, list[int]], int]:
    provenance = _object(arm.get("provenance"), "arm provenance")
    graph_contract = _object(provenance.get("graph_contract"), "provenance graph contract")
    slots = [
        _object(value, "cache slot")
        for value in _array(graph_contract.get("cache_slots"), "provenance cache slots")
    ]
    _require(bool(slots), "provenance cache slot contract is empty")
    config = _object(arm.get("config"), "arm config")
    d_model = _positive_integer(config.get("d_model"), "config d_model")
    n_heads = _positive_integer(config.get("n_heads"), "config n_heads")
    n_kv_heads = _positive_integer(config.get("n_kv_heads"), "config n_kv_heads")
    conv_kernel = _positive_integer(config.get("conv_kernel"), "config conv_kernel")
    n_layers = _positive_integer(config.get("n_layers"), "config n_layers")
    n_loops = _positive_integer(config.get("n_loops"), "config n_loops")
    layer_types = _array(config.get("layer_types"), "config layer_types")
    _require(
        d_model % n_heads == 0
        and n_heads % n_kv_heads == 0
        and len(layer_types) == n_layers
        and all(kind in {"attn", "conv"} for kind in layer_types),
        "config cannot define the declared cache layout",
    )
    _require(
        len(slots) == n_layers * n_loops,
        "cache-slot count differs from config layers and loops",
    )
    precision = arm.get("precision")
    expected_dtype = "float16" if precision == "fp16" else "float32"
    dtype_bytes = _DTYPE_BYTES[expected_dtype]
    names: list[str] = []
    expected_dims: dict[str, list[int]] = {}
    total = 0
    for slot_index, slot in enumerate(slots):
        _exact_keys(
            slot,
            {
                "dtype_by_precision",
                "kind",
                "layer",
                "loop",
                "past_inputs",
                "present_outputs",
                "shape",
                "slot",
                "update",
            },
            f"cache slot {slot_index}",
        )
        expected_loop = slot_index // n_layers
        expected_layer = slot_index % n_layers
        expected_kind = layer_types[expected_layer]
        past_inputs = _array(slot.get("past_inputs"), "cache past inputs")
        present_outputs = _array(slot.get("present_outputs"), "cache present outputs")
        _require(
            all(
                isinstance(name, str) and bool(name)
                for name in [*past_inputs, *present_outputs]
            ),
            "cache past/present names are invalid",
        )
        expected_past = (
            [f"past_{slot_index}_key", f"past_{slot_index}_value"]
            if expected_kind == "attn"
            else [f"past_{slot_index}_conv"]
        )
        expected_present = (
            [f"present_{slot_index}_key", f"present_{slot_index}_value"]
            if expected_kind == "attn"
            else [f"present_{slot_index}_conv"]
        )
        symbolic_shape: list[Any]
        concrete_shape: list[int]
        expected_update: str
        if expected_kind == "attn":
            symbolic_shape = [
                "batch",
                n_kv_heads,
                "cache_sequence",
                d_model // n_heads,
            ]
            concrete_shape = [1, n_kv_heads, attention_sequence, d_model // n_heads]
            expected_update = "append_one_token_along_axis_2"
        else:
            symbolic_shape = ["batch", d_model, conv_kernel - 1]
            concrete_shape = [1, d_model, conv_kernel - 1]
            expected_update = "replace_with_latest_fixed_width_tail"
        dtype_by_precision = _object(
            slot.get("dtype_by_precision"),
            "cache dtype-by-precision",
        )
        declared_shape = _array(slot.get("shape"), "cache symbolic shape")
        shape_matches = len(declared_shape) == len(symbolic_shape) and all(
            (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
            if isinstance(expected, int)
            else isinstance(observed, str) and observed == expected
            for observed, expected in zip(declared_shape, symbolic_shape, strict=True)
        )
        _require(
            slot.get("slot") == slot_index
            and isinstance(slot.get("slot"), int)
            and not isinstance(slot.get("slot"), bool)
            and slot.get("loop") == expected_loop
            and isinstance(slot.get("loop"), int)
            and not isinstance(slot.get("loop"), bool)
            and slot.get("layer") == expected_layer
            and isinstance(slot.get("layer"), int)
            and not isinstance(slot.get("layer"), bool)
            and slot.get("kind") == expected_kind
            and past_inputs == expected_past
            and present_outputs == expected_present
            and shape_matches
            and slot.get("update") == expected_update
            and dtype_by_precision.get(precision) == expected_dtype,
            f"cache slot {slot_index} differs from its exact config-derived contract",
        )
        names.extend(present_outputs)
        expected_dims.update({name: concrete_shape for name in present_outputs})
        per_tensor = math.prod(concrete_shape) * dtype_bytes
        total += len(present_outputs) * per_tensor
    _require(len(set(names)) == len(names), "cache present output names are duplicated")
    return names, expected_dtype, expected_dims, total


def _validate_disposal(record: Mapping[str, Any], *, cache_tensor_count: int) -> None:
    _require(
        record.get("disposal_contract_verified") is True,
        "record disposal contract was not positively verified after timing",
    )
    tracker = _object(record.get("allocation_disposal"), "record allocation disposal")
    _exact_keys(tracker, _DISPOSAL_COUNTER_KEYS, "record allocation disposal")
    decode_passes = ACCEPTANCE_OUTPUT_TOKENS - 1
    expected = {
        "cache_tensors_allocated": cache_tensor_count * ACCEPTANCE_OUTPUT_TOKENS,
        "cache_dispose_attempted": cache_tensor_count * ACCEPTANCE_OUTPUT_TOKENS,
        "cache_dispose_succeeded": cache_tensor_count * ACCEPTANCE_OUTPUT_TOKENS,
        "cache_dispose_failed": 0,
        "cache_dispose_api_unavailable": 0,
        "superseded_cache_tensors_released": cache_tensor_count * decode_passes,
        "final_cache_tensors_released": cache_tensor_count,
        "next_token_tensors_allocated": ACCEPTANCE_OUTPUT_TOKENS,
        "next_token_dispose_attempted": ACCEPTANCE_OUTPUT_TOKENS,
        "next_token_dispose_succeeded": ACCEPTANCE_OUTPUT_TOKENS,
        "next_token_dispose_failed": 0,
        "next_token_dispose_api_unavailable": 0,
        "logits_tensors_allocated": ACCEPTANCE_OUTPUT_TOKENS,
        "logits_dispose_attempted": ACCEPTANCE_OUTPUT_TOKENS,
        "logits_dispose_succeeded": ACCEPTANCE_OUTPUT_TOKENS,
        "logits_dispose_failed": 0,
        "logits_dispose_api_unavailable": 0,
        "decode_input_tensors_allocated": decode_passes,
        "decode_input_dispose_attempted": decode_passes,
        "decode_input_dispose_succeeded": decode_passes,
        "decode_input_dispose_failed": 0,
        "decode_input_dispose_api_unavailable": 0,
    }
    _require(
        all(
            isinstance(tracker.get(field), int)
            and not isinstance(tracker.get(field), bool)
            and tracker.get(field) == value
            for field, value in expected.items()
        ),
        "record disposal accounting is incomplete or contradictory",
    )


def _validate_record(
    record: Mapping[str, Any],
    *,
    phase: str,
    arm: Mapping[str, Any],
    vocab_size: int,
    benchmark_session_id: str,
    run_id: str,
    run_challenge: str,
) -> None:
    _exact_keys(record, _RECORD_KEYS, f"{phase} record")
    context = _positive_integer(record.get("input_tokens"), "record input tokens")
    _require(context in ACCEPTANCE_CONTEXT_LENGTHS, "record context is outside the protocol")
    _require(
        record.get("benchmark_session_id") == benchmark_session_id
        and record.get("run_id") == run_id
        and record.get("run_challenge") == run_challenge
        and record.get("phase") == phase
        and record.get("arm_id") == arm.get("id")
        and record.get("run_ok") is True
        and record.get("error") is None,
        "record is not a successful run for the accepted arm/phase",
    )
    _validate_no_capability_claims(record, "record")
    _require(
        record.get("decision_output_abi") == ACCEPTANCE_DECISION_ABI,
        "record did not use the final-logits decision ABI",
    )
    expected_token_counts = {
        "prompt_tokens_requested": context,
        "actual_input_tokens": context,
        "prompt_tokens_actual": context,
        "output_tokens_requested": ACCEPTANCE_OUTPUT_TOKENS,
        "actual_output_tokens": ACCEPTANCE_OUTPUT_TOKENS,
    }
    _require(
        all(
            isinstance(record.get(field), int)
            and not isinstance(record.get(field), bool)
            and record.get(field) == expected
            for field, expected in expected_token_counts.items()
        ),
        "record token counts do not match the acceptance contract",
    )
    generated = _array(record.get("generated_token_ids"), "generated token IDs")
    _require(len(generated) == ACCEPTANCE_OUTPUT_TOKENS, "generated token count mismatch")
    _require(
        all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < vocab_size
            for token in generated
        ),
        "generated token IDs are outside the model vocabulary",
    )
    passes = _object(record.get("graph_pass_counts"), "record graph pass counts")
    _exact_keys(passes, _GRAPH_PASS_KEYS, "record graph pass counts")
    expected_passes = {
        "prefill": 1,
        "decode": ACCEPTANCE_OUTPUT_TOKENS - 1,
        "prefill_attempted": 1,
        "decode_attempted": ACCEPTANCE_OUTPUT_TOKENS - 1,
        "total": ACCEPTANCE_OUTPUT_TOKENS,
        "total_attempted": ACCEPTANCE_OUTPUT_TOKENS,
        "expected_prefill": 1,
        "expected_decode": ACCEPTANCE_OUTPUT_TOKENS - 1,
        "expected_total": ACCEPTANCE_OUTPUT_TOKENS,
    }
    _require(
        all(
            isinstance(passes.get(key), int)
            and not isinstance(passes.get(key), bool)
            and passes.get(key) == value
            for key, value in expected_passes.items()
        ),
        "record graph-pass counts do not match one prefill plus 31 cached decodes",
    )
    actual_positions = _positive_integer(
        record.get("actual_graph_input_token_positions"),
        "record actual graph input token positions",
    )
    _require(
        actual_positions == context + ACCEPTANCE_OUTPUT_TOKENS - 1,
        "record graph position accounting mismatch",
    )
    graph_sha256 = _object(record.get("graph_sha256"), "record graph SHA-256")
    graph_bytes = _object(record.get("graph_bytes"), "record graph bytes")
    graph_files = _object(record.get("graph_files"), "record graph files")
    for graph_identity, label in (
        (graph_sha256, "record graph SHA-256"),
        (graph_bytes, "record graph bytes"),
        (graph_files, "record graph files"),
    ):
        _exact_keys(graph_identity, {"decode", "prefill"}, label)
    record_prefill_bytes = _positive_integer(
        graph_bytes.get("prefill"),
        "record prefill graph bytes",
    )
    record_decode_bytes = _positive_integer(
        graph_bytes.get("decode"),
        "record decode graph bytes",
    )
    _require(
        graph_sha256.get("prefill") == arm.get("prefill_sha256")
        and graph_sha256.get("decode") == arm.get("decode_sha256")
        and record_prefill_bytes == arm.get("prefill_bytes")
        and record_decode_bytes == arm.get("decode_bytes")
        and graph_files.get("prefill") == arm.get("prefill_file")
        and graph_files.get("decode") == arm.get("decode_file"),
        "record graph identities differ from the accepted arm",
    )
    decode_records = _array(record.get("decode_pass_records"), "decode pass records")
    _require(
        len(decode_records) == ACCEPTANCE_OUTPUT_TOKENS - 1,
        "decode pass-record cardinality mismatch",
    )
    cache = _object(record.get("cache"), "record cache")
    _exact_keys(cache, _CACHE_KEYS, "record cache")
    _require(
        cache.get("enabled") is True
        and cache.get("requested_residency") == "gpu-buffer"
        and cache.get("next_token_residency") == "cpu"
        and cache.get("logits_residency") == "cpu"
        and cache.get("token_selection_source") == "validated_logits_argmax"
        and cache.get("next_token_role") == "compatibility_cross_check"
        and cache.get("cache_data_read_to_javascript") is False
        and cache.get("update_strategy")
        == "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization",
        "record does not prove the required cache-bearing WebGPU path",
    )
    cache_tensor_count = _positive_integer(cache.get("tensor_count"), "cache tensor count")
    cache_slot_count = _positive_integer(cache.get("slot_count"), "cache slot count")
    declared_cache_slots = _array(cache.get("slots"), "record cache slots")
    provenance_cache_slots = _array(
        _object(
            _object(arm.get("provenance"), "arm provenance").get("graph_contract"),
            "provenance graph contract",
        ).get("cache_slots"),
        "provenance cache slots",
    )
    _require(
        cache_slot_count == len(provenance_cache_slots)
        and declared_cache_slots == provenance_cache_slots,
        "record cache slots differ from graph provenance",
    )
    (
        cache_names,
        cache_dtype,
        prefill_dims,
        expected_prefill_bytes,
    ) = _cache_contract_identity(
        arm,
        attention_sequence=context,
    )
    _require(
        cache.get("dtype") == cache_dtype,
        "record cache dtype differs from graph precision",
    )
    _require(
        cache_tensor_count == len(cache_names),
        "record cache tensor count differs from graph provenance",
    )
    prefill_count, prefill_bytes = _tensor_logical_bytes(
        cache.get("prefill_tensors"),
        "record.cache.prefill_tensors",
        expected_names=cache_names,
        expected_dtype=cache_dtype,
        expected_dims=prefill_dims,
    )
    _, _, final_dims, expected_final_bytes = _cache_contract_identity(
        arm,
        attention_sequence=context + ACCEPTANCE_OUTPUT_TOKENS - 1,
    )
    final_count, final_bytes = _tensor_logical_bytes(
        cache.get("final_tensors"),
        "record.cache.final_tensors",
        expected_names=cache_names,
        expected_dtype=cache_dtype,
        expected_dims=final_dims,
    )
    _require(
        prefill_count == cache_tensor_count == final_count,
        "cache tensor cardinality differs across prefill/final state",
    )
    _require(
        _positive_integer(
            cache.get("prefill_logical_bytes"),
            "record prefill cache logical bytes",
        )
        == prefill_bytes
        and _positive_integer(
            cache.get("final_logical_bytes"),
            "record final cache logical bytes",
        )
        == final_bytes,
        "cache logical-byte totals do not match tensor metadata",
    )
    _require(
        prefill_bytes == expected_prefill_bytes,
        "prefill cache bytes do not match graph/config algebra",
    )
    _require(
        final_bytes == expected_final_bytes,
        "final cache bytes do not match graph/config algebra",
    )

    inference_values: list[float] = []
    token_available_values: list[float] = []
    previous_after: int | None = None
    previous_token_offset = 0.0
    final_token_offset = 0.0
    for pass_index, raw_pass in enumerate(decode_records):
        decode_pass = _object(raw_pass, "decode pass record")
        _exact_keys(decode_pass, _DECODE_PASS_KEYS, "decode pass record")
        observed_integer_fields = {
            "pass_index": decode_pass.get("pass_index"),
            "input_token_id": decode_pass.get("input_token_id"),
            "output_token_id": decode_pass.get("output_token_id"),
            "input_tokens": decode_pass.get("input_tokens"),
            "output_tokens": decode_pass.get("output_tokens"),
            "attention_cache_sequence_length": decode_pass.get(
                "attention_cache_sequence_length"
            ),
            "cache_tensor_count": decode_pass.get("cache_tensor_count"),
        }
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in observed_integer_fields.values()
            ),
            "decode pass counts and indexes must be real integers",
        )
        _require(
            decode_pass.get("pass_index") == pass_index
            and decode_pass.get("input_token_id") == generated[pass_index]
            and decode_pass.get("output_token_id") == generated[pass_index + 1]
            and decode_pass.get("input_tokens") == 1
            and decode_pass.get("output_tokens") == 1
            and decode_pass.get("attention_cache_sequence_length") == context + pass_index + 1
            and decode_pass.get("cache_bound_directly_without_readback") is True
            and decode_pass.get("cache_tensor_count") == cache_tensor_count
            and decode_pass.get("cache_reported_locations") == ["gpu-buffer"],
            "decode pass does not prove iterative one-token GPU-cache rebinding",
        )
        before = _positive_integer(
            decode_pass.get("cache_logical_bytes_before"),
            "decode pass cache bytes before",
        )
        after = _positive_integer(
            decode_pass.get("cache_logical_bytes_after"),
            "decode pass cache bytes after",
        )
        _, _, pass_dims, expected_after = _cache_contract_identity(
            arm,
            attention_sequence=context + pass_index + 1,
        )
        pass_count, pass_bytes = _tensor_logical_bytes(
            decode_pass.get("cache_tensors"),
            "decode pass cache tensors",
            expected_names=cache_names,
            expected_dtype=cache_dtype,
            expected_dims=pass_dims,
        )
        _require(
            before == (prefill_bytes if previous_after is None else previous_after),
            "decode pass cache-byte chain is discontinuous",
        )
        _require(
            after == expected_after
            and pass_bytes == expected_after
            and pass_count == cache_tensor_count,
            "decode pass cache bytes or exact tensor shapes fail graph/config algebra",
        )
        previous_after = after
        inference_ms = _finite_number(
            decode_pass.get("inference_ms"),
            "decode pass inference time",
            positive=True,
        )
        token_available_ms = _finite_number(
            decode_pass.get("token_available_ms"),
            "decode pass token-available time",
            positive=True,
        )
        pass_started_offset = _finite_number(
            decode_pass.get("pass_started_offset_ms"),
            "decode pass start offset",
        )
        pass_resolved_offset = _finite_number(
            decode_pass.get("pass_resolved_offset_ms"),
            "decode pass resolution offset",
            positive=True,
        )
        token_available_offset = _finite_number(
            decode_pass.get("token_available_offset_ms"),
            "decode pass token-available offset",
            positive=True,
        )
        _require(
            pass_started_offset + 1e-9 >= previous_token_offset
            and pass_resolved_offset + 1e-9 >= pass_started_offset
            and token_available_offset + 1e-9 >= pass_resolved_offset,
            "decode pass monotonic boundaries overlap or run backwards",
        )
        _isclose(
            inference_ms,
            pass_resolved_offset - pass_started_offset,
            "decode pass inference interval",
        )
        _isclose(
            token_available_ms,
            token_available_offset - pass_started_offset,
            "decode pass token-available interval",
        )
        inference_values.append(inference_ms)
        token_available_values.append(token_available_ms)
        previous_token_offset = token_available_offset
        final_token_offset = token_available_offset
    _require(previous_after == final_bytes, "final cache bytes differ from the last decode pass")

    decode_inference_ms = _finite_number(
        record.get("decode_inference_ms"),
        "record.decode_inference_ms",
        positive=True,
    )
    decode_wall_ms = _finite_number(
        record.get("decode_wall_ms"),
        "record.decode_wall_ms",
        positive=True,
    )
    tpot_ms = _finite_number(record.get("tpot_ms"), "record.tpot_ms", positive=True)
    decode_rate = _finite_number(
        record.get("decode_tokens_per_second"),
        "record.decode_tokens_per_second",
        positive=True,
    )
    model_rate = _finite_number(
        record.get("model_decode_tokens_per_second"),
        "record.model_decode_tokens_per_second",
        positive=True,
    )
    _isclose(decode_inference_ms, sum(inference_values), "summed decode inference time")
    _isclose(decode_wall_ms, final_token_offset, "decode wall final-token boundary")
    _isclose(
        tpot_ms,
        decode_wall_ms / (ACCEPTANCE_OUTPUT_TOKENS - 1),
        "TPOT",
    )
    _isclose(decode_rate, 1000.0 / tpot_ms, "wall-clock decode rate")
    _isclose(
        model_rate,
        (ACCEPTANCE_OUTPUT_TOKENS - 1) * 1000.0 / decode_inference_ms,
        "model decode rate",
    )
    _require(
        decode_wall_ms + 1e-9 >= decode_inference_ms,
        "decode wall time cannot be shorter than summed session inference",
    )
    _require(
        decode_wall_ms + 1e-9 >= sum(token_available_values),
        "decode wall time cannot be shorter than summed token-available intervals",
    )
    prefill_ms = _finite_number(record.get("prefill_ms"), "record.prefill_ms", positive=True)
    ttft_ms = _finite_number(record.get("ttft_ms"), "record.ttft_ms", positive=True)
    generation_wall_ms = _finite_number(
        record.get("generation_wall_ms"),
        "record.generation_wall_ms",
        positive=True,
    )
    _require(ttft_ms + 1e-9 >= prefill_ms, "TTFT cannot precede prefill promise resolution")
    _isclose(generation_wall_ms, ttft_ms + decode_wall_ms, "generation wall time")
    _validate_disposal(record, cache_tensor_count=cache_tensor_count)
    _require(
        record.get("provider_requested") == "webgpu"
        and record.get("provider_actual") is None
        and record.get("graph_wide_provider_verified") is False
        and record.get("whole_session_provider_retry") is False
        and record.get("per_node_placement_verified") is False
        and record.get("per_node_fallback_status") == "unknown",
        "record provider evidence overstates or contradicts observed WebGPU evidence",
    )


def _validate_record_collection(
    values: Any,
    *,
    phase: str,
    repetitions: int,
    arm: Mapping[str, Any],
    vocab_size: int,
    benchmark_session_id: str,
    run_id: str,
    run_challenge: str,
) -> list[Mapping[str, Any]]:
    records = [
        _object(value, f"{phase} record")
        for value in _array(values, f"{phase} records")
    ]
    expected_count = len(ACCEPTANCE_CONTEXT_LENGTHS) * repetitions
    _require(len(records) == expected_count, f"{phase} record cardinality mismatch")
    expected = _expected_record_schedule(
        phase=phase,
        repetitions=repetitions,
        arm_id=arm.get("id"),
    )
    observed: list[tuple[int, int, Any, int]] = []
    for index, record in enumerate(records):
        repetition = _nonnegative_integer(
            record.get("repetition"),
            f"{phase} record {index} repetition",
        )
        order_index = _nonnegative_integer(
            record.get("order_index"),
            f"{phase} record {index} order index",
        )
        context = _positive_integer(
            record.get("input_tokens"),
            f"{phase} record {index} input tokens",
        )
        observed.append((repetition, order_index, record.get("arm_id"), context))
    _require(observed == expected, f"{phase} seeded schedule differs from the protocol")
    global_indexes = [
        _nonnegative_integer(
            record.get("global_order_index"),
            f"{phase} global order index",
        )
        for record in records
    ]
    _require(
        global_indexes == list(range(expected_count)),
        f"{phase} global order is incomplete or duplicated",
    )
    for record in records:
        _validate_record(
            record,
            phase=phase,
            arm=arm,
            vocab_size=vocab_size,
            benchmark_session_id=benchmark_session_id,
            run_id=run_id,
            run_challenge=run_challenge,
        )
    return records


def _validate_sessions(
    result: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    benchmark_session_id: str,
    run_challenge: str,
) -> None:
    sessions = [
        _object(value, "session record")
        for value in _array(result.get("session_records"), "session records")
    ]
    _require(len(sessions) == 2, "single acceptance result must contain exactly two sessions")
    provenance = _object(arm.get("provenance"), "arm provenance")
    graph_contract = _object(provenance.get("graph_contract"), "provenance graph contract")
    cache_slots = [
        _object(value, "provenance cache slot")
        for value in _array(graph_contract.get("cache_slots"), "provenance cache slots")
    ]
    past_names = [
        name
        for slot in cache_slots
        for name in _array(slot.get("past_inputs"), "cache past inputs")
    ]
    present_names = [
        name
        for slot in cache_slots
        for name in _array(slot.get("present_outputs"), "cache present outputs")
    ]
    expected_outputs = ["next_token", "logits", *present_names]
    preferred_locations = {
        "next_token": "cpu",
        "logits": "cpu",
        **dict.fromkeys(present_names, "gpu-buffer"),
    }
    expected_order = [
        graph_kind
        for _arm_id, graph_kind in _shuffled(
            [(arm.get("id"), "prefill"), (arm.get("id"), "decode")],
            ACCEPTANCE_SESSION_ORDER_SEED,
        )
    ]
    observed_order: list[str] = []
    for order_index, session in enumerate(sessions):
        _exact_keys(session, _SESSION_KEYS, "session record")
        _require(
            _nonnegative_integer(
                session.get("order_index"),
                "session order index",
            )
            == order_index,
            "session order indexes are not exact and contiguous",
        )
        graph_kind = session["graph_kind"]
        observed_order.append(graph_kind)
        session_create_ms = _finite_number(
            session.get("session_create_ms"),
            "session creation time",
            positive=True,
        )
        _require(session_create_ms > 0, "session creation time must be positive")
        graph_bytes = _positive_integer(
            session.get("graph_bytes"),
            "session graph bytes",
        )
        preferred_output_location = _object(
            session.get("preferred_output_location"),
            "session preferred output location",
        )
        _exact_keys(
            preferred_output_location,
            set(preferred_locations),
            "session preferred output location",
        )
        expected_inputs = ["input_ids"] if graph_kind == "prefill" else ["input_ids", *past_names]
        _require(
            session.get("benchmark_session_id") == benchmark_session_id
            and session.get("run_challenge") == run_challenge
            and session.get("phase") == "session_create"
            and session.get("arm_id") == arm.get("id")
            and session.get("graph_sha256") == arm.get(f"{graph_kind}_sha256")
            and graph_bytes == arm.get(f"{graph_kind}_bytes")
            and session.get("provider_requested") == "webgpu"
            and session.get("provider_actual") is None
            and session.get("execution_provider_list") == ["webgpu"]
            and session.get("exact_provider_request_and_session_creation_observed") is True
            and session.get("graph_wide_provider_verified") is False
            and session.get("per_node_placement_verified") is False
            and session.get("per_node_placement_status") == "unknown"
            and session.get("per_node_fallback_status") == "unknown"
            and session.get("whole_session_provider_retry") is False
            and session.get("cache_residency_requested") == "gpu-buffer"
            and session.get("next_token_residency_requested") == "cpu"
            and session.get("logits_residency_requested") == "cpu"
            and session.get("input_names") == expected_inputs
            and session.get("output_names") == expected_outputs
            and preferred_output_location == preferred_locations
            and session.get("error") is None,
            "session record does not match the observed WebGPU request/session contract",
        )
        ort_webgpu = _validate_ort_webgpu_evidence(
            session.get("ort_webgpu"),
            "session ORT WebGPU evidence",
        )
        _require(
            ort_webgpu.get("ort_adapter_available") is True
            and ort_webgpu.get("ort_device_available") is True,
            "session record lacks ORT WebGPU adapter/device evidence",
        )
    _require(
        observed_order == expected_order,
        "session creation order differs from the seeded acceptance schedule",
    )


def _validate_protocol(result: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(result, _RAW_RESULT_KEYS, "browser result")
    _require(
        _positive_integer(result.get("schema_version"), "browser result schema version") == 1,
        "unsupported browser result schema",
    )
    _require(result.get("benchmark") == ACCEPTANCE_BENCHMARK, "wrong acceptance benchmark kind")
    _require(result.get("status") == "complete", "browser benchmark is not complete")
    _validate_no_capability_claims(result, "browser result", require_artifact=True)
    _require(
        result.get("trained_weights") is True
        and result.get("untrained_random_weights") is False
        and result.get("latency_only") is True
        and result.get("action_capability_claimed") is False
        and result.get("quality_evaluation") is False
        and result.get("quality_scored_separately") is True,
        "browser result labels are not trained latency-only acceptance labels",
    )
    _require(not result.get("errors") and not result.get("failures"), "result contains failures")
    _utc_timestamp(result.get("created_at"), "browser result created_at")
    metadata = _object(result.get("metadata"), "metadata")
    _exact_keys(metadata, _METADATA_KEYS, "metadata")
    _validate_no_capability_claims(metadata, "metadata")
    _uuid4(metadata.get("benchmark_session_id"), "metadata benchmark session ID")
    _uuid4(metadata.get("run_id"), "metadata run ID")
    _require(
        metadata.get("benchmark_session_id") != metadata.get("run_id"),
        "benchmark session and run identities must differ",
    )
    _sha256(metadata.get("run_challenge"), "metadata run challenge")
    _sha256(
        metadata.get("external_machine_condition_sha256"),
        "metadata external machine condition SHA-256",
    )
    _require(
        metadata.get("evidence_scope") == EVIDENCE_SCOPE,
        "browser result evidence scope is missing or overstated",
    )
    _require(
        metadata.get("benchmark_mode") == "single"
        and metadata.get("artifact_mode") == "trained"
        and metadata.get("acceptance_mode") is True
        and metadata.get("decision_output_abi") == ACCEPTANCE_DECISION_ABI,
        "result is not an explicit single-model final-logits acceptance run",
    )
    acceptance = _object(metadata.get("acceptance_protocol"), "acceptance protocol")
    expected = {
        "id": ACCEPTANCE_PROTOCOL_ID,
        "context_lengths": list(ACCEPTANCE_CONTEXT_LENGTHS),
        "output_tokens_per_condition": ACCEPTANCE_OUTPUT_TOKENS,
        "warmups_per_condition": ACCEPTANCE_WARMUPS,
        "measured_repetitions_per_condition": ACCEPTANCE_REPETITIONS,
        "case_order_seed": ACCEPTANCE_SEED,
        "exact": True,
    }
    for field in (
        "output_tokens_per_condition",
        "warmups_per_condition",
        "measured_repetitions_per_condition",
    ):
        _positive_integer(acceptance.get(field), f"acceptance protocol {field}")
    _require(dict(acceptance) == expected, "acceptance protocol declaration is not exact")
    output_tokens = _positive_integer(
        metadata.get("output_tokens_per_condition"),
        "metadata output tokens per condition",
    )
    warmups = _positive_integer(
        metadata.get("warmups_per_condition"),
        "metadata warmups per condition",
    )
    repetitions = _positive_integer(
        metadata.get("measured_repetitions_per_condition"),
        "metadata measured repetitions per condition",
    )
    _require(
        metadata.get("context_lengths") == list(ACCEPTANCE_CONTEXT_LENGTHS)
        and metadata.get("prompt_lengths_tokens") == list(ACCEPTANCE_CONTEXT_LENGTHS)
        and output_tokens == ACCEPTANCE_OUTPUT_TOKENS
        and warmups == ACCEPTANCE_WARMUPS
        and repetitions == ACCEPTANCE_REPETITIONS
        and metadata.get("case_order_seed") == ACCEPTANCE_SEED
        and metadata.get("session_order_seed") == ACCEPTANCE_SESSION_ORDER_SEED,
        "browser protocol fields differ from the acceptance contract",
    )
    graph_contract = _object(metadata.get("graph_pass_contract"), "graph pass contract")
    _exact_keys(
        graph_contract,
        {
            "decode_per_condition",
            "first_token_source",
            "prefill_per_condition",
            "remaining_token_source",
            "total_per_condition",
        },
        "graph pass contract",
    )
    prefill_passes = _positive_integer(
        graph_contract.get("prefill_per_condition"),
        "graph prefill passes per condition",
    )
    decode_passes = _positive_integer(
        graph_contract.get("decode_per_condition"),
        "graph decode passes per condition",
    )
    total_passes = _positive_integer(
        graph_contract.get("total_per_condition"),
        "graph total passes per condition",
    )
    _require(
        prefill_passes == 1
        and decode_passes == ACCEPTANCE_OUTPUT_TOKENS - 1
        and total_passes == ACCEPTANCE_OUTPUT_TOKENS
        and graph_contract.get("first_token_source")
        == "prefill.logits argmax; next_token compatibility cross-check"
        and graph_contract.get("remaining_token_source")
        == "decode.logits argmax; next_token compatibility cross-check",
        "declared graph-pass contract is not one prefill plus 31 logits-driven decodes",
    )
    cache_contract = _object(metadata.get("cache_contract"), "cache contract")
    _exact_keys(
        cache_contract,
        {
            "cache_data_read_to_javascript",
            "enabled",
            "logits_residency",
            "next_token_residency",
            "superseded_and_final_cache_disposal_attempted",
            "token_selection_source",
            "update_strategy",
            "wasm_cache_residency",
            "webgpu_cache_residency",
        },
        "cache contract",
    )
    _require(
        cache_contract.get("enabled") is True
        and cache_contract.get("webgpu_cache_residency") == "gpu-buffer"
        and cache_contract.get("next_token_residency") == "cpu"
        and cache_contract.get("logits_residency") == "cpu"
        and cache_contract.get("token_selection_source") == "validated_logits_argmax"
        and cache_contract.get("cache_data_read_to_javascript") is False
        and cache_contract.get("superseded_and_final_cache_disposal_attempted") is True,
        "declared cache contract is not the acceptance WebGPU path",
    )
    return metadata


def build_webgpu_decode_receipt(
    result_payload: bytes,
    *,
    expected_wrapper_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    expected_run_challenge: str,
    expected_machine_condition_sha256: str,
    expected_harness_html_sha256: str,
    expected_harness_javascript_sha256: str,
    expected_ort_javascript_sha256: str,
    expected_ort_wasm_sha256: str,
) -> dict[str, Any]:
    """Validate one raw browser result against an external root and return its receipt."""

    result = _object(
        _strict_json_bytes(result_payload, label="WebGPU decode result"),
        "WebGPU decode result",
    )
    metadata = _validate_protocol(result)
    benchmark_session_id = _uuid4(
        metadata.get("benchmark_session_id"),
        "metadata benchmark session ID",
    )
    run_id = _uuid4(metadata.get("run_id"), "metadata run ID")
    run_challenge = _external_sha256(expected_run_challenge, "expected run challenge")
    machine_condition_sha256 = _external_sha256(
        expected_machine_condition_sha256,
        "expected machine condition SHA-256",
    )
    _require(
        metadata.get("run_challenge") == run_challenge,
        "browser result run challenge differs from the external challenge",
    )
    _require(
        metadata.get("external_machine_condition_sha256") == machine_condition_sha256,
        "browser result machine condition differs from the external anchor",
    )
    harness_identity = _validate_harness_identity(
        metadata,
        expected_html_sha256=expected_harness_html_sha256,
        expected_javascript_sha256=expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=expected_ort_wasm_sha256,
    )
    manifest, manifest_sha256, manifest_bytes = _validate_manifest(
        metadata,
        expected_wrapper_manifest_sha256=expected_wrapper_manifest_sha256,
    )
    provider = _validate_provider(metadata)
    records = _artifact_records(result)
    source_record = _find_wrapper_record(
        records,
        sha256=manifest_sha256,
        size=manifest_bytes,
    )
    arms = _array(metadata.get("arms"), "metadata.arms")
    _require(
        _positive_integer(metadata.get("arm_count"), "metadata arm count") == 1
        and len(arms) == 1,
        "result must contain one arm",
    )
    arm = _object(arms[0], "metadata.arms[0]")
    (
        provenance,
        checkpoint_sha256,
        checkpoint_stage,
        training_artifacts,
        bundle_identities,
    ) = _validate_arm_provenance(arm, manifest, records)
    _require(
        checkpoint_sha256
        == _external_sha256(expected_checkpoint_sha256, "expected checkpoint SHA-256"),
        "result checkpoint differs from the accepted checkpoint",
    )
    config = _object(arm.get("config"), "arm config")
    vocab_size = _positive_integer(config.get("vocab_size"), "model vocabulary size")
    _validate_inputs(result, vocab_size=vocab_size)
    _validate_sessions(
        result,
        arm,
        benchmark_session_id=benchmark_session_id,
        run_challenge=run_challenge,
    )
    warmups = _validate_record_collection(
        result.get("warmup_records"),
        phase="warmup",
        repetitions=ACCEPTANCE_WARMUPS,
        arm=arm,
        vocab_size=vocab_size,
        benchmark_session_id=benchmark_session_id,
        run_id=run_id,
        run_challenge=run_challenge,
    )
    measured = _validate_record_collection(
        result.get("records"),
        phase="measured",
        repetitions=ACCEPTANCE_REPETITIONS,
        arm=arm,
        vocab_size=vocab_size,
        benchmark_session_id=benchmark_session_id,
        run_id=run_id,
        run_challenge=run_challenge,
    )
    summary = _object(result.get("summary"), "browser summary")
    _exact_keys(summary, _SUMMARY_KEYS, "browser summary")
    attempted = _nonnegative_integer(summary.get("attempted"), "browser summary attempted")
    completed = _nonnegative_integer(summary.get("completed"), "browser summary completed")
    failed = _nonnegative_integer(summary.get("failed"), "browser summary failed")
    _require(
        attempted == len(measured) and completed == len(measured) and failed == 0,
        "browser summary cardinalities disagree with raw records",
    )
    tokenizer = _object(provenance.get("tokenizer"), "provenance.tokenizer")
    tokenizer_sha256 = _sha256(arm.get("tokenizer_sha256"), "tokenizer SHA-256")
    _require(
        tokenizer.get("sha256") == tokenizer_sha256,
        "arm tokenizer digest differs from raw provenance",
    )
    source_identity = {
        "bytes": len(result_payload),
        "sha256": hashlib.sha256(result_payload).hexdigest(),
    }
    lineage_identity = bundle_identities["lineage"]
    receipt_without_hash: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "verified": True,
        "result": source_identity,
        "benchmark_created_at": result.get("created_at"),
        "execution": {
            "benchmark_session_id": benchmark_session_id,
            "external_machine_condition_sha256": machine_condition_sha256,
            "run_challenge": run_challenge,
            "run_id": run_id,
        },
        "protocol": {
            "id": ACCEPTANCE_PROTOCOL_ID,
            "context_lengths": list(ACCEPTANCE_CONTEXT_LENGTHS),
            "output_tokens_per_condition": ACCEPTANCE_OUTPUT_TOKENS,
            "warmups_per_condition": ACCEPTANCE_WARMUPS,
            "measured_repetitions_per_condition": ACCEPTANCE_REPETITIONS,
            "case_order_seed": ACCEPTANCE_SEED,
            "session_order_seed": ACCEPTANCE_SESSION_ORDER_SEED,
            "warmup_records": len(warmups),
            "measured_records": len(measured),
            "prefill_passes_per_record": 1,
            "decode_passes_per_record": ACCEPTANCE_OUTPUT_TOKENS - 1,
        },
        "model": {
            "name": arm.get("id"),
            "pair_role": arm.get("pair_role"),
            "checkpoint_stage": checkpoint_stage,
            "checkpoint_sha256": checkpoint_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "training_artifact_sha256": [
                identity["sha256"] for identity in training_artifacts
            ],
            "training_artifacts": training_artifacts,
        },
        "bundle": {
            "external_wrapper_manifest_sha256": manifest_sha256,
            "wrapper_manifest_file": source_record.get("relative_path"),
            "wrapper_manifest_bytes": manifest_bytes,
            "wrapper_manifest_sha256": manifest_sha256,
            "provenance_file": bundle_identities["provenance"]["file"],
            "provenance_bytes": bundle_identities["provenance"]["bytes"],
            "provenance_sha256": bundle_identities["provenance"]["sha256"],
            "training_lineage_file": lineage_identity["file"],
            "training_lineage_bytes": lineage_identity["bytes"],
            "training_lineage_sha256": lineage_identity["sha256"],
            "prefill_file": bundle_identities["prefill"]["file"],
            "prefill_bytes": bundle_identities["prefill"]["bytes"],
            "prefill_sha256": bundle_identities["prefill"]["sha256"],
            "decode_file": bundle_identities["decode"]["file"],
            "decode_bytes": bundle_identities["decode"]["bytes"],
            "decode_sha256": bundle_identities["decode"]["sha256"],
        },
        "runtime": {
            "provider_requested": provider.get("provider_requested"),
            "provider_actual": None,
            "execution_provider_list": provider.get("execution_provider_list"),
            "exact_provider_request_and_session_creation_observed": True,
            "cache_output_location_observed": "gpu-buffer",
            "graph_wide_provider_verified": False,
            "per_node_placement_verified": False,
            "per_node_placement_status": "unknown",
            "whole_session_provider_retry": False,
            "ort_version": metadata.get("ort_version_reported"),
            "browser": metadata.get("browser"),
            "gpu": metadata.get("gpu"),
            "harness_identity": harness_identity,
        },
        "record_contract": {
            "all_records_successful": True,
            "all_token_counts_verified": True,
            "all_graph_pass_counts_verified": True,
            "all_decode_cache_rebindings_verified": True,
            "all_input_fixture_hashes_verified": True,
            "all_metric_algebra_verified": True,
            "all_cache_byte_algebra_verified": True,
            "all_tensor_disposals_verified": True,
            "raw_provenance_bytes_verified": True,
            "raw_training_lineage_bytes_verified": True,
            "external_wrapper_root_verified": True,
            "external_acquisition_roots_verified": True,
            "external_machine_condition_crosschecked": True,
            "external_run_challenge_crosschecked": True,
            "source_controlled_harness_verified": True,
            "run_identity_crosschecked": True,
        },
        "metrics_by_context": _metric_summaries(measured),
        "scope": dict(EVIDENCE_SCOPE),
    }
    receipt_self_sha256 = hashlib.sha256(
        canonical_json_bytes(receipt_without_hash)
    ).hexdigest()
    receipt = {
        **receipt_without_hash,
        "receipt_self_sha256": receipt_self_sha256,
    }
    verify_webgpu_decode_receipt_bytes(canonical_json_bytes(receipt))
    return receipt


def _verify_receipt_protocol(value: Any) -> None:
    protocol = _object(value, "receipt.protocol")
    expected = {
        "id": ACCEPTANCE_PROTOCOL_ID,
        "context_lengths": list(ACCEPTANCE_CONTEXT_LENGTHS),
        "output_tokens_per_condition": ACCEPTANCE_OUTPUT_TOKENS,
        "warmups_per_condition": ACCEPTANCE_WARMUPS,
        "measured_repetitions_per_condition": ACCEPTANCE_REPETITIONS,
        "case_order_seed": ACCEPTANCE_SEED,
        "session_order_seed": ACCEPTANCE_SESSION_ORDER_SEED,
        "warmup_records": len(ACCEPTANCE_CONTEXT_LENGTHS) * ACCEPTANCE_WARMUPS,
        "measured_records": len(ACCEPTANCE_CONTEXT_LENGTHS) * ACCEPTANCE_REPETITIONS,
        "prefill_passes_per_record": 1,
        "decode_passes_per_record": ACCEPTANCE_OUTPUT_TOKENS - 1,
    }
    integer_fields = {
        "output_tokens_per_condition",
        "warmups_per_condition",
        "measured_repetitions_per_condition",
        "warmup_records",
        "measured_records",
        "prefill_passes_per_record",
        "decode_passes_per_record",
    }
    _require(
        all(
            isinstance(protocol.get(field), int) and not isinstance(protocol.get(field), bool)
            for field in integer_fields
        ),
        "receipt protocol counts must be real integers",
    )
    _require(dict(protocol) == expected, "receipt protocol is not the exact acceptance protocol")


def _verify_receipt_execution(value: Any) -> None:
    execution = _object(value, "receipt.execution")
    _exact_keys(
        execution,
        {
            "benchmark_session_id",
            "external_machine_condition_sha256",
            "run_challenge",
            "run_id",
        },
        "receipt.execution",
    )
    benchmark_session_id = _uuid4(
        execution.get("benchmark_session_id"),
        "receipt benchmark session ID",
    )
    run_id = _uuid4(execution.get("run_id"), "receipt run ID")
    _sha256(execution.get("run_challenge"), "receipt run challenge")
    _sha256(
        execution.get("external_machine_condition_sha256"),
        "receipt external machine condition SHA-256",
    )
    _require(
        benchmark_session_id != run_id,
        "receipt benchmark session and run identities must differ",
    )


def _verify_receipt_model(value: Any) -> None:
    model = _object(value, "receipt.model")
    _exact_keys(
        model,
        {
            "checkpoint_sha256",
            "checkpoint_stage",
            "name",
            "pair_role",
            "tokenizer_sha256",
            "training_artifact_sha256",
            "training_artifacts",
        },
        "receipt.model",
    )
    _require(
        isinstance(model.get("name"), str)
        and bool(model["name"])
        and isinstance(model.get("pair_role"), str)
        and bool(model["pair_role"]),
        "receipt model identity is incomplete",
    )
    stage = model.get("checkpoint_stage")
    _require(stage in {"pretrain", "midtrain", "sft", "rl"}, "receipt checkpoint stage is invalid")
    _sha256(model.get("checkpoint_sha256"), "receipt checkpoint SHA-256")
    _sha256(model.get("tokenizer_sha256"), "receipt tokenizer SHA-256")
    hashes = [
        _sha256(value, "receipt training artifact SHA-256")
        for value in _array(
            model.get("training_artifact_sha256"),
            "receipt training artifact hashes",
        )
    ]
    identities = [
        _validate_training_identity(value, index)
        for index, value in enumerate(
            _array(model.get("training_artifacts"), "receipt training artifacts")
        )
    ]
    _require(
        [identity["sha256"] for identity in identities] == hashes,
        "receipt training identity/hash set differs",
    )
    _require(
        len(set(hashes)) == len(hashes)
        and len({identity["path"] for identity in identities}) == len(identities),
        "receipt training identities are duplicated",
    )
    if stage in {"midtrain", "sft", "rl"}:
        _require(bool(identities), "posttraining receipt has no exact training artifact identity")


def _verify_receipt_bundle(value: Any, *, checkpoint_stage: str) -> None:
    bundle = _object(value, "receipt.bundle")
    _exact_keys(
        bundle,
        {
            "decode_bytes",
            "decode_file",
            "decode_sha256",
            "external_wrapper_manifest_sha256",
            "prefill_bytes",
            "prefill_file",
            "prefill_sha256",
            "provenance_bytes",
            "provenance_file",
            "provenance_sha256",
            "training_lineage_bytes",
            "training_lineage_file",
            "training_lineage_sha256",
            "wrapper_manifest_bytes",
            "wrapper_manifest_file",
            "wrapper_manifest_sha256",
        },
        "receipt.bundle",
    )
    for prefix in ("wrapper_manifest", "provenance", "prefill", "decode"):
        filename = bundle.get(f"{prefix}_file")
        _require(
            isinstance(filename, str) and bool(filename),
            f"receipt {prefix} filename is missing",
        )
        _positive_integer(bundle.get(f"{prefix}_bytes"), f"receipt {prefix} bytes")
        _sha256(bundle.get(f"{prefix}_sha256"), f"receipt {prefix} SHA-256")
    _require(
        _sha256(
            bundle.get("external_wrapper_manifest_sha256"),
            "receipt external wrapper manifest SHA-256",
        )
        == bundle.get("wrapper_manifest_sha256"),
        "receipt wrapper manifest differs from its recorded external root",
    )
    lineage_values = (
        bundle.get("training_lineage_file"),
        bundle.get("training_lineage_bytes"),
        bundle.get("training_lineage_sha256"),
    )
    if checkpoint_stage in {"midtrain", "sft", "rl"}:
        _require(
            lineage_values[0] == "training-lineage.json",
            "posttraining receipt has no canonical lineage file",
        )
        _positive_integer(lineage_values[1], "receipt training lineage bytes")
        _sha256(lineage_values[2], "receipt training lineage SHA-256")
    else:
        all_null = all(value is None for value in lineage_values)
        all_present = (
            isinstance(lineage_values[0], str)
            and bool(lineage_values[0])
            and isinstance(lineage_values[1], int)
            and not isinstance(lineage_values[1], bool)
            and lineage_values[1] > 0
            and isinstance(lineage_values[2], str)
        )
        _require(all_null or all_present, "pretrain receipt has a partial lineage identity")
        if all_present:
            _sha256(lineage_values[2], "receipt training lineage SHA-256")


def _verify_receipt_runtime(value: Any) -> None:
    runtime = _object(value, "receipt.runtime")
    _exact_keys(
        runtime,
        {
            "browser",
            "cache_output_location_observed",
            "exact_provider_request_and_session_creation_observed",
            "execution_provider_list",
            "gpu",
            "graph_wide_provider_verified",
            "harness_identity",
            "ort_version",
            "per_node_placement_status",
            "per_node_placement_verified",
            "provider_actual",
            "provider_requested",
            "whole_session_provider_retry",
        },
        "receipt.runtime",
    )
    _require(
        runtime.get("provider_requested") == "webgpu"
        and runtime.get("provider_actual") is None
        and runtime.get("execution_provider_list") == ["webgpu"]
        and runtime.get("exact_provider_request_and_session_creation_observed") is True
        and runtime.get("cache_output_location_observed") == "gpu-buffer"
        and runtime.get("graph_wide_provider_verified") is False
        and runtime.get("per_node_placement_verified") is False
        and runtime.get("per_node_placement_status") == "unknown"
        and runtime.get("whole_session_provider_retry") is False,
        "receipt runtime evidence is contradictory or overstated",
    )
    _require(
        isinstance(runtime.get("ort_version"), str) and bool(runtime["ort_version"]),
        "receipt ONNX Runtime version is unknown",
    )
    harness = _object(runtime.get("harness_identity"), "receipt.runtime.harness_identity")
    html = _object(harness.get("html"), "receipt.runtime.harness_identity.html")
    javascript = _object(
        harness.get("javascript"),
        "receipt.runtime.harness_identity.javascript",
    )
    ort = _object(harness.get("ort"), "receipt.runtime.harness_identity.ort")
    ort_javascript = _object(
        ort.get("javascript"),
        "receipt.runtime.harness_identity.ort.javascript",
    )
    ort_wasm = _object(ort.get("wasm"), "receipt.runtime.harness_identity.ort.wasm")
    _validate_harness_identity(
        {
            "harness_identity": harness,
            "acceptance_acquisition_roots": {
                "html_sha256": html.get("external_expected_sha256"),
                "javascript_sha256": javascript.get("external_expected_sha256"),
                "ort_javascript_sha256": ort_javascript.get(
                    "external_expected_sha256"
                ),
                "ort_wasm_sha256": ort_wasm.get("external_expected_sha256"),
            },
            "ort_script_url": ort_javascript.get("url"),
            "ort_wasm_url": ort_wasm.get("url"),
            "ort_version_pin": runtime.get("ort_version"),
            "ort_version_reported": runtime.get("ort_version"),
            "ort_version_verified": True,
        },
        expected_html_sha256=html.get("external_expected_sha256"),
        expected_javascript_sha256=javascript.get("external_expected_sha256"),
        expected_ort_javascript_sha256=ort_javascript.get("external_expected_sha256"),
        expected_ort_wasm_sha256=ort_wasm.get("external_expected_sha256"),
    )
    browser = _object(runtime.get("browser"), "receipt.runtime.browser")
    _exact_keys(
        browser,
        {
            "device_memory_gb",
            "hardware_concurrency",
            "language",
            "languages",
            "mobile",
            "platform",
            "user_agent",
            "user_agent_brands",
        },
        "receipt.runtime.browser",
    )
    _require(
        browser.get("user_agent") is None or isinstance(browser["user_agent"], str),
        "receipt browser user agent is invalid",
    )
    _require(
        browser.get("language") is None or isinstance(browser["language"], str),
        "receipt browser language is invalid",
    )
    _require(
        browser.get("platform") is None or isinstance(browser["platform"], str),
        "receipt browser platform is invalid",
    )
    languages = browser.get("languages")
    _require(
        languages is None
        or (
            isinstance(languages, list)
            and all(isinstance(language, str) for language in languages)
        ),
        "receipt browser languages are invalid",
    )
    brands = browser.get("user_agent_brands")
    _require(
        brands is None
        or (
            isinstance(brands, list)
            and all(
                isinstance(brand, Mapping)
                and set(brand) == {"brand", "version"}
                and isinstance(brand["brand"], str)
                and isinstance(brand["version"], str)
                for brand in brands
            )
        ),
        "receipt browser brands are invalid",
    )
    _require(
        browser.get("mobile") is None or isinstance(browser["mobile"], bool),
        "receipt browser mobile flag is invalid",
    )
    hardware_concurrency = browser.get("hardware_concurrency")
    _require(
        hardware_concurrency is None
        or (
            isinstance(hardware_concurrency, int)
            and not isinstance(hardware_concurrency, bool)
            and hardware_concurrency > 0
        ),
        "receipt browser hardware concurrency is invalid",
    )
    device_memory = browser.get("device_memory_gb")
    _require(
        device_memory is None
        or (
            isinstance(device_memory, (int, float))
            and not isinstance(device_memory, bool)
            and math.isfinite(float(device_memory))
            and float(device_memory) > 0
        ),
        "receipt browser device memory is invalid",
    )
    gpu = _object(runtime.get("gpu"), "receipt.runtime.gpu")
    _exact_keys(
        gpu,
        {
            "device_features",
            "device_label",
            "navigator_gpu_available",
            "ort_webgpu",
        },
        "receipt.runtime.gpu",
    )
    _require(
        gpu.get("navigator_gpu_available") is True,
        "receipt runtime did not observe navigator.gpu",
    )
    ort_webgpu = _object(gpu.get("ort_webgpu"), "receipt.runtime.gpu.ort_webgpu")
    _exact_keys(
        ort_webgpu,
        {"adapter_info", "ort_adapter_available", "ort_device_available"},
        "receipt.runtime.gpu.ort_webgpu",
    )
    _require(
        ort_webgpu.get("ort_adapter_available") is True
        and ort_webgpu.get("ort_device_available") is True,
        "receipt runtime lacks ORT WebGPU adapter/device evidence",
    )
    _require(
        gpu.get("device_label") is None or isinstance(gpu["device_label"], str),
        "receipt GPU device label is invalid",
    )
    adapter_info = _object(
        ort_webgpu.get("adapter_info"),
        "receipt.runtime.gpu.ort_webgpu.adapter_info",
    )
    _exact_keys(
        adapter_info,
        {
            "architecture",
            "description",
            "device",
            "is_fallback_adapter",
            "vendor",
        },
        "receipt.runtime.gpu.ort_webgpu.adapter_info",
    )
    for field in ("architecture", "description", "device", "vendor"):
        _require(
            adapter_info[field] is None or isinstance(adapter_info[field], str),
            f"receipt GPU adapter {field} is invalid",
        )
    _require(
        adapter_info["is_fallback_adapter"] is None
        or isinstance(adapter_info["is_fallback_adapter"], bool),
        "receipt GPU fallback-adapter flag is invalid",
    )
    features = gpu.get("device_features")
    _require(
        features is None
        or (
            isinstance(features, list)
            and all(isinstance(feature, str) for feature in features)
            and features == sorted(features)
        ),
        "receipt GPU feature list is invalid",
    )


def _verify_receipt_contract(value: Any) -> None:
    contract = _object(value, "receipt.record_contract")
    expected_fields = {
        "all_cache_byte_algebra_verified",
        "all_decode_cache_rebindings_verified",
        "all_graph_pass_counts_verified",
        "all_input_fixture_hashes_verified",
        "all_metric_algebra_verified",
        "all_records_successful",
        "all_tensor_disposals_verified",
        "all_token_counts_verified",
        "external_wrapper_root_verified",
        "external_acquisition_roots_verified",
        "external_machine_condition_crosschecked",
        "external_run_challenge_crosschecked",
        "raw_provenance_bytes_verified",
        "raw_training_lineage_bytes_verified",
        "run_identity_crosschecked",
        "source_controlled_harness_verified",
    }
    _exact_keys(contract, expected_fields, "receipt.record_contract")
    _require(
        all(contract[field] is True for field in expected_fields),
        "receipt record contract is not wholly verified",
    )


def _verify_receipt_metrics(value: Any) -> None:
    rows = _array(value, "receipt.metrics_by_context")
    _require(len(rows) == len(ACCEPTANCE_CONTEXT_LENGTHS), "receipt metric row count mismatch")
    expected_metric_keys = {*_METRICS, "final_logical_cache_bytes"}
    for index, (row_value, context) in enumerate(zip(rows, ACCEPTANCE_CONTEXT_LENGTHS)):
        row = _object(row_value, f"receipt.metrics_by_context[{index}]")
        _exact_keys(row, {"input_tokens", "metrics"}, f"receipt metric row {index}")
        _require(row.get("input_tokens") == context, "receipt metric contexts are not exact")
        metrics = _object(row.get("metrics"), f"receipt metric row {index}.metrics")
        _exact_keys(metrics, expected_metric_keys, f"receipt metric row {index}.metrics")
        for metric_name, summary_value in metrics.items():
            summary = _object(summary_value, f"receipt metric {metric_name}")
            _exact_keys(summary, {"p50", "p95"}, f"receipt metric {metric_name}")
            p50 = _finite_number(summary.get("p50"), f"receipt {metric_name}.p50", positive=True)
            p95 = _finite_number(summary.get("p95"), f"receipt {metric_name}.p95", positive=True)
            _require(p95 >= p50, f"receipt {metric_name} percentile order is invalid")


def verify_webgpu_decode_receipt_bytes(payload: bytes) -> dict[str, Any]:
    """Verify canonical serialization, exact schema, semantics, and receipt self-hash."""

    receipt = _object(
        _strict_json_bytes(payload, label="WebGPU decode receipt"),
        "WebGPU decode receipt",
    )
    _require(payload == canonical_json_bytes(receipt), "receipt is not canonical sorted JSON")
    _exact_keys(receipt, _RECEIPT_KEYS, "receipt")
    unsigned = dict(receipt)
    declared = _sha256(unsigned.pop("receipt_self_sha256", None), "receipt self SHA-256")
    _require(
        declared == hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        "receipt self SHA-256 mismatch",
    )
    _require(
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and receipt.get("kind") == RECEIPT_KIND
        and receipt.get("verified") is True,
        "receipt identity is unsupported",
    )
    _utc_timestamp(receipt.get("benchmark_created_at"), "receipt benchmark timestamp")
    result = _object(receipt.get("result"), "receipt.result")
    _exact_keys(result, {"bytes", "sha256"}, "receipt.result")
    _positive_integer(result.get("bytes"), "receipt result bytes")
    _sha256(result.get("sha256"), "receipt result SHA-256")
    _verify_receipt_protocol(receipt.get("protocol"))
    _verify_receipt_execution(receipt.get("execution"))
    _verify_receipt_model(receipt.get("model"))
    model = _object(receipt["model"], "receipt.model")
    _verify_receipt_bundle(
        receipt.get("bundle"),
        checkpoint_stage=str(model["checkpoint_stage"]),
    )
    _verify_receipt_runtime(receipt.get("runtime"))
    _verify_receipt_contract(receipt.get("record_contract"))
    _verify_receipt_metrics(receipt.get("metrics_by_context"))
    _require(receipt.get("scope") == EVIDENCE_SCOPE, "receipt evidence scope is overstated")
    return dict(receipt)


def write_webgpu_decode_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically publish a new canonical receipt without replacing any path."""

    payload = canonical_json_bytes(dict(receipt))
    verify_webgpu_decode_receipt_bytes(payload)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite WebGPU decode receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite concurrently created WebGPU decode receipt: {path}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    _require(
        read_stable_webgpu_evidence_file(
            path,
            label="published WebGPU decode receipt",
        )
        == payload,
        "published WebGPU decode receipt failed byte verification",
    )
