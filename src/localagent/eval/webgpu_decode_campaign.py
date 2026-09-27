"""Fail-closed three-run campaign gate for single-model WebGPU cached decode."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.eval.webgpu_decode_receipt import (
    ACCEPTANCE_CONTEXT_LENGTHS,
    EVIDENCE_SCOPE,
    build_webgpu_decode_receipt,
    read_stable_webgpu_evidence_file,
    verify_webgpu_decode_receipt_bytes,
)


CAMPAIGN_KIND = "localagent_single_webgpu_cached_decode_acceptance_campaign"
CAMPAIGN_SCHEMA_VERSION = 2
CAMPAIGN_RUNS = 3
MIN_P50_DECODE_TOKENS_PER_SECOND = 100.0
MAX_P95_TPOT_MS = 10.0
EXPECTED_WARMUP_RECORDS = 36
EXPECTED_MEASURED_RECORDS = 360
EXPECTED_GRAPH_CALLS = 12_672
_SHA256_LENGTH = 64
_CAMPAIGN_KEYS = {
    "acceptance_gate",
    "campaign_self_sha256",
    "common_condition",
    "counts",
    "external_anchors",
    "kind",
    "receipts",
    "schema_version",
    "scope",
    "verified",
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


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _external_sha256(value: Any, label: str) -> str:
    digest = _sha256(value, label)
    _require(digest != "0" * _SHA256_LENGTH, f"{label} must not be the zero digest")
    return digest


def _positive_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be a finite number",
    )
    return float(value)


def _timestamp(value: Any, label: str) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), f"{label} must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is invalid") from error
    _require(parsed.utcoffset() is not None, f"{label} must be timezone-aware")
    return parsed


def _median_of_three(values: Sequence[float], label: str) -> float:
    _require(len(values) == CAMPAIGN_RUNS, f"{label} requires exactly three values")
    return sorted(values)[1]


def _file_identity(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _normalized_trace_sha256(raw_payload: bytes) -> str:
    try:
        raw = _object(json.loads(raw_payload), "raw result")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("raw result is not valid UTF-8 JSON") from error

    def project_record(record: Any) -> dict[str, Any]:
        value = _object(record, "raw trace record")
        fields = (
            "phase",
            "global_order_index",
            "repetition",
            "order_index",
            "arm_id",
            "input_tokens",
            "generated_token_ids",
            "graph_pass_counts",
            "actual_graph_input_token_positions",
            "ttft_ms",
            "prefill_ms",
            "decode_inference_ms",
            "decode_wall_ms",
            "generation_wall_ms",
            "tpot_ms",
            "decode_tokens_per_second",
            "model_decode_tokens_per_second",
            "decode_pass_records",
            "cache",
            "allocation_disposal",
        )
        return {field: copy.deepcopy(value[field]) for field in fields}

    def project_session(record: Any) -> dict[str, Any]:
        value = _object(record, "raw trace session")
        fields = (
            "order_index",
            "arm_id",
            "graph_kind",
            "session_create_ms",
            "graph_sha256",
            "graph_bytes",
            "input_names",
            "output_names",
        )
        return {field: copy.deepcopy(value[field]) for field in fields}

    def project_input(record: Any) -> dict[str, Any]:
        value = _object(record, "raw trace input")
        fields = (
            "input_tokens",
            "actual_tensor_tokens",
            "input_ids_int64_sha256",
            "tensor_dtype",
            "tensor_dims",
        )
        return {field: copy.deepcopy(value[field]) for field in fields}

    normalized = {
        "inputs": [
            project_input(record) for record in _array(raw.get("inputs"), "raw inputs")
        ],
        "session_records": [
            project_session(record)
            for record in _array(raw.get("session_records"), "raw session records")
        ],
        "warmup_records": [
            project_record(record)
            for record in _array(raw.get("warmup_records"), "raw warmup records")
        ],
        "records": [
            project_record(record)
            for record in _array(raw.get("records"), "raw measured records")
        ],
    }
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _context_gate(receipt: Mapping[str, Any], *, run_index: int) -> list[dict[str, Any]]:
    rows = _array(receipt.get("metrics_by_context"), f"receipt {run_index} metrics")
    _require(
        len(rows) == len(ACCEPTANCE_CONTEXT_LENGTHS),
        f"receipt {run_index} metric row count differs",
    )
    output: list[dict[str, Any]] = []
    for row_index, expected_context in enumerate(ACCEPTANCE_CONTEXT_LENGTHS):
        row = _object(rows[row_index], f"receipt {run_index} metric row {row_index}")
        _require(
            isinstance(row.get("input_tokens"), int)
            and not isinstance(row.get("input_tokens"), bool)
            and row.get("input_tokens") == expected_context,
            f"receipt {run_index} context order differs",
        )
        metrics = _object(row.get("metrics"), f"receipt {run_index} context metrics")
        decode = _object(
            metrics.get("decode_tokens_per_second"),
            f"receipt {run_index} decode rate",
        )
        tpot = _object(metrics.get("tpot_ms"), f"receipt {run_index} TPOT")
        p50_decode = _finite_number(
            decode.get("p50"),
            f"receipt {run_index} context {expected_context} p50 decode rate",
        )
        p95_tpot = _finite_number(
            tpot.get("p95"),
            f"receipt {run_index} context {expected_context} p95 TPOT",
        )
        decode_passed = p50_decode >= MIN_P50_DECODE_TOKENS_PER_SECOND
        tpot_passed = p95_tpot <= MAX_P95_TPOT_MS
        _require(
            decode_passed and tpot_passed,
            f"receipt {run_index} context {expected_context} misses the latency gate",
        )
        output.append(
            {
                "input_tokens": expected_context,
                "p50_decode_tokens_per_second": p50_decode,
                "p95_tpot_ms": p95_tpot,
                "p50_decode_tokens_per_second_passed": decode_passed,
                "p95_tpot_ms_passed": tpot_passed,
                "passed": True,
            }
        )
    return output


def _common_condition(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = receipts[0]
    fields = ("model", "bundle", "protocol", "runtime", "scope")
    first_execution = _object(first.get("execution"), "receipt 1 execution")
    machine_anchor = first_execution.get("external_machine_condition_sha256")
    for run_index, receipt in enumerate(receipts[1:], start=2):
        for field in fields:
            _require(
                receipt.get(field) == first.get(field),
                f"receipt {run_index} {field} differs across the campaign",
            )
        execution = _object(receipt.get("execution"), f"receipt {run_index} execution")
        _require(
            execution.get("external_machine_condition_sha256") == machine_anchor,
            f"receipt {run_index} external machine condition differs",
        )
    return {
        **{field: copy.deepcopy(first[field]) for field in fields},
        "external_machine_condition_sha256": machine_anchor,
    }


def _campaign_core(
    receipt_payloads: Sequence[bytes],
    receipts: Sequence[Mapping[str, Any]],
    raw_file_identities: Sequence[Mapping[str, Any]],
    normalized_trace_hashes: Sequence[str],
    *,
    expected_checkpoint_sha256: str,
    expected_wrapper_manifest_sha256: str,
    expected_run_challenges: Sequence[str],
    expected_machine_condition_sha256: str,
    expected_harness_html_sha256: str,
    expected_harness_javascript_sha256: str,
    expected_ort_javascript_sha256: str,
    expected_ort_wasm_sha256: str,
) -> dict[str, Any]:
    _require(
        len(receipt_payloads)
        == len(receipts)
        == len(raw_file_identities)
        == len(normalized_trace_hashes)
        == CAMPAIGN_RUNS,
        "campaign requires exactly three raw-result/receipt pairs",
    )
    checkpoint_sha256 = _external_sha256(
        expected_checkpoint_sha256,
        "expected checkpoint SHA-256",
    )
    wrapper_sha256 = _external_sha256(
        expected_wrapper_manifest_sha256,
        "expected wrapper-manifest SHA-256",
    )
    run_challenges = [
        _external_sha256(value, f"expected run challenge {index}")
        for index, value in enumerate(expected_run_challenges, start=1)
    ]
    _require(
        len(run_challenges) == CAMPAIGN_RUNS
        and len(set(run_challenges)) == CAMPAIGN_RUNS,
        "campaign requires exactly three distinct external run challenges",
    )
    machine_anchor = _external_sha256(
        expected_machine_condition_sha256,
        "expected machine condition SHA-256",
    )
    acquisition_roots = {
        "harness_html_sha256": _external_sha256(
            expected_harness_html_sha256,
            "expected harness HTML SHA-256",
        ),
        "harness_javascript_sha256": _external_sha256(
            expected_harness_javascript_sha256,
            "expected harness JavaScript SHA-256",
        ),
        "ort_javascript_sha256": _external_sha256(
            expected_ort_javascript_sha256,
            "expected ORT JavaScript SHA-256",
        ),
        "ort_wasm_sha256": _external_sha256(
            expected_ort_wasm_sha256,
            "expected ORT WASM SHA-256",
        ),
    }

    timestamps: list[datetime] = []
    raw_result_hashes: list[str] = []
    receipt_file_hashes: list[str] = []
    receipt_self_hashes: list[str] = []
    session_ids: list[str] = []
    run_ids: list[str] = []
    per_run_gate: list[dict[str, Any]] = []
    receipt_entries: list[dict[str, Any]] = []
    for run_index, (payload, receipt, raw_identity, trace_hash, challenge) in enumerate(
        zip(
            receipt_payloads,
            receipts,
            raw_file_identities,
            normalized_trace_hashes,
            run_challenges,
            strict=True,
        ),
        start=1,
    ):
        _require(
            payload == canonical_json_bytes(dict(receipt)),
            f"receipt {run_index} bytes are not canonical",
        )
        raw_identity = _object(raw_identity, f"raw result {run_index} file identity")
        _exact_keys(raw_identity, {"bytes", "sha256"}, "raw-result file identity")
        raw_bytes = _positive_integer(raw_identity.get("bytes"), "raw-result file bytes")
        raw_sha256 = _sha256(raw_identity.get("sha256"), "raw-result file SHA-256")
        trace_sha256 = _sha256(trace_hash, "normalized execution trace SHA-256")
        model = _object(receipt.get("model"), f"receipt {run_index} model")
        bundle = _object(receipt.get("bundle"), f"receipt {run_index} bundle")
        execution = _object(receipt.get("execution"), f"receipt {run_index} execution")
        result = _object(receipt.get("result"), f"receipt {run_index} result identity")
        _require(
            model.get("checkpoint_sha256") == checkpoint_sha256,
            f"receipt {run_index} checkpoint differs from the external anchor",
        )
        _require(
            bundle.get("external_wrapper_manifest_sha256") == wrapper_sha256
            and bundle.get("wrapper_manifest_sha256") == wrapper_sha256,
            f"receipt {run_index} wrapper manifest differs from the external anchor",
        )
        _require(
            execution.get("run_challenge") == challenge,
            f"receipt {run_index} challenge differs from the predetermined external challenge",
        )
        _require(
            execution.get("external_machine_condition_sha256") == machine_anchor,
            f"receipt {run_index} machine condition differs from the external anchor",
        )
        _require(
            result.get("bytes") == raw_bytes
            and result.get("sha256") == raw_sha256,
            f"receipt {run_index} result identity differs from raw file bytes",
        )
        created_at = receipt.get("benchmark_created_at")
        timestamps.append(_timestamp(created_at, f"receipt {run_index} timestamp"))
        raw_result_hashes.append(raw_sha256)
        session_id = execution.get("benchmark_session_id")
        run_id = execution.get("run_id")
        _require(
            isinstance(session_id, str) and isinstance(run_id, str),
            f"receipt {run_index} execution identities are missing",
        )
        session_ids.append(session_id)
        run_ids.append(run_id)
        receipt_self_hash = _sha256(
            receipt.get("receipt_self_sha256"),
            f"receipt {run_index} self SHA-256",
        )
        receipt_file_sha256 = hashlib.sha256(payload).hexdigest()
        receipt_file_hashes.append(receipt_file_sha256)
        receipt_self_hashes.append(receipt_self_hash)
        contexts = _context_gate(receipt, run_index=run_index)
        per_run_gate.append(
            {
                "run_index": run_index,
                "benchmark_created_at": created_at,
                "benchmark_session_id": session_id,
                "run_id": run_id,
                "run_challenge": challenge,
                "raw_result_sha256": raw_sha256,
                "normalized_execution_trace_sha256": trace_sha256,
                "contexts": contexts,
                "passed": True,
            }
        )
        receipt_entries.append(
            {
                "run_index": run_index,
                "raw_result_file": {"bytes": raw_bytes, "sha256": raw_sha256},
                "normalized_execution_trace_sha256": trace_sha256,
                "receipt_file": {
                    "bytes": len(payload),
                    "sha256": receipt_file_sha256,
                },
                "receipt": copy.deepcopy(dict(receipt)),
            }
        )

    _require(
        all(left < right for left, right in zip(timestamps, timestamps[1:])),
        "campaign receipt timestamps must be distinct and strictly chronological",
    )
    _require(
        len(set(raw_result_hashes)) == CAMPAIGN_RUNS,
        "campaign raw-result SHA-256 identities must be distinct",
    )
    _require(
        len(set(normalized_trace_hashes)) == CAMPAIGN_RUNS,
        "campaign normalized execution traces must be distinct",
    )
    _require(
        len(set(receipt_file_hashes)) == CAMPAIGN_RUNS
        and len(set(receipt_self_hashes)) == CAMPAIGN_RUNS,
        "campaign receipt identities must be distinct",
    )
    _require(
        len(set(session_ids)) == CAMPAIGN_RUNS
        and len(set(run_ids)) == CAMPAIGN_RUNS
        and len(set(session_ids + run_ids)) == CAMPAIGN_RUNS * 2,
        "campaign benchmark session/run identities must all be distinct",
    )

    common = _common_condition(receipts)
    protocols = [_object(receipt["protocol"], "receipt protocol") for receipt in receipts]
    warmup_records = sum(
        _positive_integer(protocol.get("warmup_records"), "receipt warmup records")
        for protocol in protocols
    )
    measured_records = sum(
        _positive_integer(protocol.get("measured_records"), "receipt measured records")
        for protocol in protocols
    )
    total_records = warmup_records + measured_records
    prefill_graph_calls = sum(
        (
            _positive_integer(protocol.get("warmup_records"), "receipt warmup records")
            + _positive_integer(protocol.get("measured_records"), "receipt measured records")
        )
        * _positive_integer(protocol.get("prefill_passes_per_record"), "receipt prefill passes")
        for protocol in protocols
    )
    decode_graph_calls = sum(
        (
            _positive_integer(protocol.get("warmup_records"), "receipt warmup records")
            + _positive_integer(protocol.get("measured_records"), "receipt measured records")
        )
        * _positive_integer(protocol.get("decode_passes_per_record"), "receipt decode passes")
        for protocol in protocols
    )
    graph_calls = prefill_graph_calls + decode_graph_calls
    warmup_graph_calls = sum(
        _positive_integer(protocol.get("warmup_records"), "receipt warmup records")
        * (
            _positive_integer(protocol.get("prefill_passes_per_record"), "receipt prefill passes")
            + _positive_integer(protocol.get("decode_passes_per_record"), "receipt decode passes")
        )
        for protocol in protocols
    )
    measured_graph_calls = sum(
        _positive_integer(protocol.get("measured_records"), "receipt measured records")
        * (
            _positive_integer(protocol.get("prefill_passes_per_record"), "receipt prefill passes")
            + _positive_integer(protocol.get("decode_passes_per_record"), "receipt decode passes")
        )
        for protocol in protocols
    )
    _require(
        warmup_records == EXPECTED_WARMUP_RECORDS
        and measured_records == EXPECTED_MEASURED_RECORDS
        and graph_calls == EXPECTED_GRAPH_CALLS,
        "campaign accounting does not equal 36 warmups, 360 measurements, and 12672 graph calls",
    )

    median_gate: list[dict[str, Any]] = []
    for row_index, context in enumerate(ACCEPTANCE_CONTEXT_LENGTHS):
        decode_values = [
            _finite_number(
                run["contexts"][row_index]["p50_decode_tokens_per_second"],
                f"context {context} campaign p50 decode value",
            )
            for run in per_run_gate
        ]
        tpot_values = [
            _finite_number(
                run["contexts"][row_index]["p95_tpot_ms"],
                f"context {context} campaign p95 TPOT value",
            )
            for run in per_run_gate
        ]
        median_decode = _median_of_three(decode_values, f"context {context} decode median")
        median_tpot = _median_of_three(tpot_values, f"context {context} TPOT median")
        decode_passed = median_decode >= MIN_P50_DECODE_TOKENS_PER_SECOND
        tpot_passed = median_tpot <= MAX_P95_TPOT_MS
        _require(
            decode_passed and tpot_passed,
            f"context {context} median-of-three misses the latency gate",
        )
        median_gate.append(
            {
                "input_tokens": context,
                "median_p50_decode_tokens_per_second": median_decode,
                "median_p95_tpot_ms": median_tpot,
                "p50_decode_tokens_per_second_passed": decode_passed,
                "p95_tpot_ms_passed": tpot_passed,
                "passed": True,
            }
        )

    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "kind": CAMPAIGN_KIND,
        "verified": True,
        "external_anchors": {
            "checkpoint_sha256": checkpoint_sha256,
            "wrapper_manifest_sha256": wrapper_sha256,
            "run_challenges": run_challenges,
            "machine_condition_sha256": machine_anchor,
            **acquisition_roots,
        },
        "common_condition": common,
        "receipts": receipt_entries,
        "counts": {
            "runs": CAMPAIGN_RUNS,
            "warmup_records": warmup_records,
            "measured_records": measured_records,
            "total_records": total_records,
            "warmup_graph_calls": warmup_graph_calls,
            "measured_graph_calls": measured_graph_calls,
            "prefill_graph_calls": prefill_graph_calls,
            "decode_graph_calls": decode_graph_calls,
            "graph_calls": graph_calls,
        },
        "acceptance_gate": {
            "thresholds": {
                "minimum_per_context_p50_decode_tokens_per_second": (
                    MIN_P50_DECODE_TOKENS_PER_SECOND
                ),
                "maximum_per_context_p95_tpot_ms": MAX_P95_TPOT_MS,
                "aggregation": "median_of_three_run_level_context_statistics",
            },
            "authoritative_verification": (
                "requires_exact_saved_campaign_plus_three_raw_and_receipt_sidecar_pairs"
            ),
            "artifact_backed_verification_required": True,
            "raw_receipts_rebuilt_and_byte_compared": True,
            "distinct_external_run_challenges": True,
            "distinct_normalized_execution_traces": True,
            "distinct_raw_result_sha256": True,
            "distinct_receipt_identities": True,
            "distinct_benchmark_session_and_run_ids": True,
            "strictly_chronological_distinct_timestamps": True,
            "per_run": per_run_gate,
            "median_of_three_by_context": median_gate,
            "passed": True,
        },
        "scope": dict(EVIDENCE_SCOPE),
    }


def build_webgpu_decode_campaign(
    run_pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    expected_checkpoint_sha256: str,
    expected_wrapper_manifest_sha256: str,
    expected_run_challenges: Sequence[str],
    expected_machine_condition_sha256: str,
    expected_harness_html_sha256: str,
    expected_harness_javascript_sha256: str,
    expected_ort_javascript_sha256: str,
    expected_ort_wasm_sha256: str,
) -> dict[str, Any]:
    """Rebuild three receipts from stably read raw/receipt pairs, then gate the campaign."""

    _require(
        len(run_pairs) == CAMPAIGN_RUNS,
        "campaign requires exactly three raw-result/receipt path pairs",
    )
    _require(
        len(expected_run_challenges) == CAMPAIGN_RUNS,
        "campaign requires exactly three predetermined run challenges",
    )
    raw_payloads: list[bytes] = []
    receipt_payloads: list[bytes] = []
    receipts: list[dict[str, Any]] = []
    raw_identities: list[dict[str, Any]] = []
    trace_hashes: list[str] = []
    for run_index, (raw_path, receipt_path) in enumerate(run_pairs, start=1):
        raw_payload = read_stable_webgpu_evidence_file(
            raw_path,
            label=f"run {run_index} raw result",
        )
        receipt_payload = read_stable_webgpu_evidence_file(
            receipt_path,
            label=f"run {run_index} receipt",
        )
        rebuilt = build_webgpu_decode_receipt(
            raw_payload,
            expected_wrapper_manifest_sha256=expected_wrapper_manifest_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_run_challenge=expected_run_challenges[run_index - 1],
            expected_machine_condition_sha256=expected_machine_condition_sha256,
            expected_harness_html_sha256=expected_harness_html_sha256,
            expected_harness_javascript_sha256=expected_harness_javascript_sha256,
            expected_ort_javascript_sha256=expected_ort_javascript_sha256,
            expected_ort_wasm_sha256=expected_ort_wasm_sha256,
        )
        verified = verify_webgpu_decode_receipt_bytes(receipt_payload)
        _require(
            receipt_payload == canonical_json_bytes(rebuilt)
            and verified == rebuilt,
            f"run {run_index} receipt is not the exact receipt rebuilt from raw bytes",
        )
        raw_payloads.append(raw_payload)
        receipt_payloads.append(receipt_payload)
        receipts.append(verified)
        raw_identities.append(_file_identity(raw_payload))
        trace_hashes.append(_normalized_trace_sha256(raw_payload))
    _require(len(raw_payloads) == CAMPAIGN_RUNS, "campaign input collection is incomplete")
    core = _campaign_core(
        receipt_payloads,
        receipts,
        raw_identities,
        trace_hashes,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_wrapper_manifest_sha256=expected_wrapper_manifest_sha256,
        expected_run_challenges=expected_run_challenges,
        expected_machine_condition_sha256=expected_machine_condition_sha256,
        expected_harness_html_sha256=expected_harness_html_sha256,
        expected_harness_javascript_sha256=expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=expected_ort_wasm_sha256,
    )
    campaign = {
        **core,
        "campaign_self_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }
    verify_webgpu_decode_campaign_integrity_bytes(canonical_json_bytes(campaign))
    return campaign


def verify_webgpu_decode_campaign_integrity_bytes(payload: bytes) -> dict[str, Any]:
    """Check only the canonical campaign envelope and embedded receipt consistency.

    This is intentionally not an acceptance assertion. It has no raw-result sidecars and
    therefore cannot establish that embedded receipts were rebuilt from raw evidence.
    """

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("WebGPU decode campaign is not valid UTF-8 JSON") from error
    campaign = _object(value, "WebGPU decode campaign")
    _require(payload == canonical_json_bytes(campaign), "campaign is not canonical sorted JSON")
    _exact_keys(campaign, _CAMPAIGN_KEYS, "campaign")
    unsigned = dict(campaign)
    declared = _sha256(unsigned.pop("campaign_self_sha256", None), "campaign self SHA-256")
    _require(
        declared == hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        "campaign self SHA-256 mismatch",
    )
    _require(
        campaign.get("schema_version") == CAMPAIGN_SCHEMA_VERSION
        and campaign.get("kind") == CAMPAIGN_KIND
        and campaign.get("verified") is True,
        "campaign identity is unsupported",
    )
    _require(campaign.get("scope") == EVIDENCE_SCOPE, "campaign evidence scope is overstated")
    anchors = _object(campaign.get("external_anchors"), "campaign external anchors")
    _exact_keys(
        anchors,
        {
            "checkpoint_sha256",
            "wrapper_manifest_sha256",
            "run_challenges",
            "machine_condition_sha256",
            "harness_html_sha256",
            "harness_javascript_sha256",
            "ort_javascript_sha256",
            "ort_wasm_sha256",
        },
        "campaign external anchors",
    )
    run_challenges = [
        _sha256(value, "campaign run challenge")
        for value in _array(anchors.get("run_challenges"), "campaign run challenges")
    ]
    entries = _array(campaign.get("receipts"), "campaign receipts")
    _require(len(entries) == CAMPAIGN_RUNS, "campaign requires exactly three embedded receipts")
    receipt_payloads: list[bytes] = []
    receipts: list[dict[str, Any]] = []
    raw_identities: list[dict[str, Any]] = []
    trace_hashes: list[str] = []
    for run_index, entry_value in enumerate(entries, start=1):
        entry = _object(entry_value, f"campaign receipt entry {run_index}")
        _exact_keys(
            entry,
            {
                "normalized_execution_trace_sha256",
                "raw_result_file",
                "receipt",
                "receipt_file",
                "run_index",
            },
            "campaign receipt entry",
        )
        _require(
            isinstance(entry.get("run_index"), int)
            and not isinstance(entry.get("run_index"), bool)
            and entry.get("run_index") == run_index,
            "campaign receipt order is not canonical",
        )
        receipt = _object(entry.get("receipt"), f"campaign embedded receipt {run_index}")
        receipt_payload = canonical_json_bytes(receipt)
        verified = verify_webgpu_decode_receipt_bytes(receipt_payload)
        file_identity = _object(
            entry.get("receipt_file"),
            f"campaign receipt file identity {run_index}",
        )
        _exact_keys(file_identity, {"bytes", "sha256"}, "campaign receipt file identity")
        _require(
            _positive_integer(file_identity.get("bytes"), "campaign receipt bytes")
            == len(receipt_payload)
            and _sha256(file_identity.get("sha256"), "campaign receipt file SHA-256")
            == hashlib.sha256(receipt_payload).hexdigest(),
            f"campaign receipt file identity {run_index} differs from embedded bytes",
        )
        raw_identity = _object(
            entry.get("raw_result_file"),
            f"campaign raw-result identity {run_index}",
        )
        _exact_keys(raw_identity, {"bytes", "sha256"}, "campaign raw-result identity")
        raw_identities.append(
            {
                "bytes": _positive_integer(
                    raw_identity.get("bytes"),
                    "campaign raw-result bytes",
                ),
                "sha256": _sha256(
                    raw_identity.get("sha256"),
                    "campaign raw-result SHA-256",
                ),
            }
        )
        trace_hashes.append(
            _sha256(
                entry.get("normalized_execution_trace_sha256"),
                "campaign normalized execution trace SHA-256",
            )
        )
        receipt_payloads.append(receipt_payload)
        receipts.append(verified)
    expected = _campaign_core(
        receipt_payloads,
        receipts,
        raw_identities,
        trace_hashes,
        expected_checkpoint_sha256=_sha256(
            anchors.get("checkpoint_sha256"),
            "campaign checkpoint anchor",
        ),
        expected_wrapper_manifest_sha256=_sha256(
            anchors.get("wrapper_manifest_sha256"),
            "campaign wrapper-manifest anchor",
        ),
        expected_run_challenges=run_challenges,
        expected_machine_condition_sha256=_sha256(
            anchors.get("machine_condition_sha256"),
            "campaign machine-condition anchor",
        ),
        expected_harness_html_sha256=_sha256(
            anchors.get("harness_html_sha256"),
            "campaign harness HTML anchor",
        ),
        expected_harness_javascript_sha256=_sha256(
            anchors.get("harness_javascript_sha256"),
            "campaign harness JavaScript anchor",
        ),
        expected_ort_javascript_sha256=_sha256(
            anchors.get("ort_javascript_sha256"),
            "campaign ORT JavaScript anchor",
        ),
        expected_ort_wasm_sha256=_sha256(
            anchors.get("ort_wasm_sha256"),
            "campaign ORT WASM anchor",
        ),
    )
    _require(unsigned == expected, "campaign fields differ from independently recomputed evidence")
    return dict(campaign)


def verify_webgpu_decode_campaign_against_artifacts(
    campaign_path: str | Path,
    run_pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    expected_checkpoint_sha256: str,
    expected_wrapper_manifest_sha256: str,
    expected_run_challenges: Sequence[str],
    expected_machine_condition_sha256: str,
    expected_harness_html_sha256: str,
    expected_harness_javascript_sha256: str,
    expected_ort_javascript_sha256: str,
    expected_ort_wasm_sha256: str,
) -> dict[str, Any]:
    """Authoritatively verify a saved campaign against all exact raw/receipt sidecars."""

    payload = read_stable_webgpu_evidence_file(
        campaign_path,
        label="saved WebGPU decode campaign",
    )
    campaign = verify_webgpu_decode_campaign_integrity_bytes(payload)
    rebuilt = build_webgpu_decode_campaign(
        run_pairs,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_wrapper_manifest_sha256=expected_wrapper_manifest_sha256,
        expected_run_challenges=expected_run_challenges,
        expected_machine_condition_sha256=expected_machine_condition_sha256,
        expected_harness_html_sha256=expected_harness_html_sha256,
        expected_harness_javascript_sha256=expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=expected_ort_wasm_sha256,
    )
    _require(
        payload == canonical_json_bytes(rebuilt) and campaign == rebuilt,
        "saved campaign does not exactly match artifact-backed reconstruction",
    )
    return campaign


def write_webgpu_decode_campaign(
    path: str | Path,
    campaign: Mapping[str, Any],
    run_pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    expected_checkpoint_sha256: str,
    expected_wrapper_manifest_sha256: str,
    expected_run_challenges: Sequence[str],
    expected_machine_condition_sha256: str,
    expected_harness_html_sha256: str,
    expected_harness_javascript_sha256: str,
    expected_ort_javascript_sha256: str,
    expected_ort_wasm_sha256: str,
) -> None:
    """Publish only a campaign that exactly reconstructs from external artifacts."""

    destination = Path(path)
    payload = canonical_json_bytes(dict(campaign))
    verify_webgpu_decode_campaign_integrity_bytes(payload)
    rebuilt = build_webgpu_decode_campaign(
        run_pairs,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_wrapper_manifest_sha256=expected_wrapper_manifest_sha256,
        expected_run_challenges=expected_run_challenges,
        expected_machine_condition_sha256=expected_machine_condition_sha256,
        expected_harness_html_sha256=expected_harness_html_sha256,
        expected_harness_javascript_sha256=expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=expected_ort_wasm_sha256,
    )
    _require(
        payload == canonical_json_bytes(rebuilt),
        "campaign cannot be published without exact artifact-backed reconstruction",
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite WebGPU decode campaign: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite concurrently created WebGPU decode campaign: {destination}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    verify_webgpu_decode_campaign_against_artifacts(
        destination,
        run_pairs,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_wrapper_manifest_sha256=expected_wrapper_manifest_sha256,
        expected_run_challenges=expected_run_challenges,
        expected_machine_condition_sha256=expected_machine_condition_sha256,
        expected_harness_html_sha256=expected_harness_html_sha256,
        expected_harness_javascript_sha256=expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=expected_ort_wasm_sha256,
    )
