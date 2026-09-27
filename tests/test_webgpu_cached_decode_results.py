from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/paper/results/m5-webgpu-cached-decode-20260728.summary.json"
ARMS = ("webgpu-35m-attn", "webgpu-35m-hybrid")
CONTEXT_LENGTHS = (128, 512, 1024, 1536)
RAW_METRICS = {
    "ttft_ms": "ttft_ms",
    "tpot_ms": "tpot_ms",
    "wall_decode_tokens_per_second": "decode_tokens_per_second",
    "model_only_decode_tokens_per_second": "model_decode_tokens_per_second",
    "final_logical_cache_bytes": "cache.final_logical_bytes",
}
CACHE_TENSORS = {
    "webgpu-35m-attn": 24,
    "webgpu-35m-hybrid": 16,
}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_value(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    return float(value)


def _expected_cache_bytes(arm: str, token_positions: int) -> int:
    attention_bytes_per_position = (
        24 if arm == "webgpu-35m-attn" else 8
    ) * 64 * 2
    fixed_conv_bytes = 0 if arm == "webgpu-35m-attn" else 8 * 448 * 2 * 2
    return attention_bytes_per_position * token_positions + fixed_conv_bytes


def _assert_provider_evidence(evidence: dict[str, Any]) -> None:
    assert evidence["provider_requested"] == "webgpu"
    assert evidence["provider_actual"] == "webgpu"
    assert evidence["execution_provider_list"] == ["webgpu"]
    assert evidence["whole_session_provider_retry"] is False
    assert evidence["provider_verified"] is None
    assert evidence["per_node_placement_verified"] is False
    assert evidence["per_node_placement_status"] == "unknown"
    assert evidence["per_node_fallback_status"] == "unknown"
    assert evidence["ort_webgpu"] == {
        "ort_adapter_available": False,
        "ort_device_available": True,
        "adapter_info": None,
    }


def _assert_session_evidence(run: dict[str, Any]) -> None:
    sessions = run["session_records"]
    assert len(sessions) == 4
    assert {(session["arm_id"], session["graph_kind"]) for session in sessions} == {
        (arm, graph_kind) for arm in ARMS for graph_kind in ("prefill", "decode")
    }
    for session in sessions:
        assert session["error"] is None
        assert session["provider_requested"] == session["provider_actual"] == "webgpu"
        assert session["execution_provider_list"] == ["webgpu"]
        assert session["whole_session_provider_retry"] is False
        assert session["per_node_placement_verified"] is False
        assert session["per_node_fallback_status"] == "unknown"
        assert session["cache_residency_requested"] == "gpu-buffer"
        assert session["next_token_residency_requested"] == "cpu"
        locations = session["preferred_output_location"]
        assert locations["next_token"] == "cpu"
        assert all(
            location == "gpu-buffer"
            for name, location in locations.items()
            if name.startswith("present_")
        )
        assert set(locations) == set(session["output_names"])
        assert session["latency_only"] is True
        assert session["untrained_random_weights"] is True
        assert session["capability_artifact"] is False
        assert session["quality_evaluation"] is False


def _assert_disposal_evidence(record: dict[str, Any]) -> None:
    tensor_count = CACHE_TENSORS[record["arm_id"]]
    disposal = record["allocation_disposal"]
    expected = {
        "cache_tensors_allocated": tensor_count * 32,
        "cache_dispose_attempted": tensor_count * 32,
        "cache_dispose_succeeded": tensor_count * 32,
        "cache_dispose_failed": 0,
        "cache_dispose_api_unavailable": 0,
        "next_token_tensors_allocated": 32,
        "next_token_dispose_attempted": 32,
        "next_token_dispose_succeeded": 32,
        "next_token_dispose_failed": 0,
        "next_token_dispose_api_unavailable": 0,
        "decode_input_tensors_allocated": 31,
        "decode_input_dispose_attempted": 31,
        "decode_input_dispose_succeeded": 31,
        "decode_input_dispose_failed": 0,
        "decode_input_dispose_api_unavailable": 0,
        "superseded_cache_tensors_released": tensor_count * 31,
        "final_cache_tensors_released": tensor_count,
    }
    assert disposal == expected


def _assert_pass_and_cache_evidence(record: dict[str, Any]) -> None:
    input_tokens = record["input_tokens"]
    assert record["output_tokens_requested"] == record["actual_output_tokens"] == 32
    assert record["actual_input_tokens"] == input_tokens
    assert record["actual_graph_input_token_positions"] == input_tokens + 31
    assert record["graph_pass_counts"] == {
        "prefill": 1,
        "decode": 31,
        "prefill_attempted": 1,
        "decode_attempted": 31,
        "total": 32,
        "total_attempted": 32,
        "expected_prefill": 1,
        "expected_decode": 31,
        "expected_total": 32,
    }
    assert len(record["generated_token_ids"]) == 32
    token_availability_overhead_ms = record["ttft_ms"] - record["prefill_ms"]
    assert 0.0 <= token_availability_overhead_ms <= 0.2

    cache = record["cache"]
    assert cache["enabled"] is True
    assert cache["dtype"] == "float16"
    assert cache["requested_residency"] == "gpu-buffer"
    assert cache["next_token_residency"] == "cpu"
    assert (
        cache["update_strategy"]
        == "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization"
    )
    assert cache["cache_data_read_to_javascript"] is False
    assert cache["tensor_count"] == CACHE_TENSORS[record["arm_id"]]
    assert cache["slot_count"] == 12
    assert len(cache["prefill_tensors"]) == cache["tensor_count"]
    assert len(cache["final_tensors"]) == cache["tensor_count"]
    assert all(
        tensor["requested_location"] == tensor["reported_location"] == "gpu-buffer"
        for tensor in cache["prefill_tensors"]
    )
    assert all(
        tensor["reported_location"] == "gpu-buffer" for tensor in cache["final_tensors"]
    )
    assert cache["prefill_logical_bytes"] == sum(
        tensor["logical_bytes"] for tensor in cache["prefill_tensors"]
    )
    assert cache["final_logical_bytes"] == sum(
        tensor["logical_bytes"] for tensor in cache["final_tensors"]
    )
    assert cache["prefill_logical_bytes"] == _expected_cache_bytes(
        record["arm_id"], input_tokens
    )
    assert cache["final_logical_bytes"] == _expected_cache_bytes(
        record["arm_id"], input_tokens + 31
    )

    passes = record["decode_pass_records"]
    assert len(passes) == 31
    assert [decode_pass["pass_index"] for decode_pass in passes] == list(range(31))
    assert record["decode_inference_ms"] == pytest.approx(
        sum(decode_pass["inference_ms"] for decode_pass in passes)
    )
    summed_per_pass_token_availability_ms = sum(
        decode_pass["token_available_ms"] for decode_pass in passes
    )
    wall_decode_ms = record["tpot_ms"] * 31
    assert wall_decode_ms >= summed_per_pass_token_availability_ms
    assert record["decode_tokens_per_second"] == pytest.approx(31_000 / wall_decode_ms)
    assert record["model_decode_tokens_per_second"] == pytest.approx(
        31_000 / record["decode_inference_ms"]
    )
    for index, decode_pass in enumerate(passes):
        assert decode_pass["input_tokens"] == decode_pass["output_tokens"] == 1
        assert decode_pass["attention_cache_sequence_length"] == input_tokens + index + 1
        assert decode_pass["cache_tensor_count"] == cache["tensor_count"]
        assert decode_pass["cache_reported_locations"] == ["gpu-buffer"]
        assert decode_pass["cache_bound_directly_without_readback"] is True
        assert decode_pass["token_available_ms"] >= decode_pass["inference_ms"]
        assert decode_pass["input_token_id"] == record["generated_token_ids"][index]
        assert decode_pass["output_token_id"] == record["generated_token_ids"][index + 1]
        expected_before = (
            cache["prefill_logical_bytes"]
            if index == 0
            else passes[index - 1]["cache_logical_bytes_after"]
        )
        assert decode_pass["cache_logical_bytes_before"] == expected_before
        assert decode_pass["cache_logical_bytes_after"] == _expected_cache_bytes(
            record["arm_id"], input_tokens + index + 1
        )
    assert passes[-1]["cache_logical_bytes_after"] == cache["final_logical_bytes"]

    _assert_disposal_evidence(record)


def _assert_record_labels(record: dict[str, Any], phase: str) -> None:
    assert record["phase"] == phase
    assert record["run_ok"] is True
    assert record["error"] is None
    assert record["provider_requested"] == record["provider_actual"] == "webgpu"
    assert record["whole_session_provider_retry"] is False
    assert record["per_node_placement_verified"] is False
    assert record["per_node_fallback_status"] == "unknown"
    assert record["latency_only"] is True
    assert record["untrained_random_weights"] is True
    assert record["capability_artifact"] is False
    assert record["quality_evaluation"] is False


def test_tracked_cached_decode_raw_artifacts_reproduce_the_summary() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    per_run: list[dict[tuple[str, int, str], dict[str, float]]] = []

    assert summary["aggregate"]["pooling"] == "none"
    assert summary["aggregate"]["reported_point_estimate"] == (
        "median of the three within-run percentiles"
    )
    assert summary["aggregate"]["reported_range"] == (
        "minimum and maximum of the three within-run percentiles"
    )
    assert summary["aggregate"]["paired_ratio_estimator"] == (
        "median and full range of three within-run hybrid_over_attention ratios"
    )
    assert summary["runtime"]["onnxruntime_web_pin"] == "1.27.0"
    assert summary["runtime"]["onnxruntime_web_reported"] == "1.27.0"
    assert summary["runtime"]["onnxruntime_version_verified"] is True
    assert summary["runtime"]["execution_provider_request"] == ["webgpu"]
    assert summary["runtime"]["execution_provider_actual"] == "webgpu"
    assert summary["runtime"]["whole_session_provider_retry"] is False
    assert summary["runtime"]["per_node_placement_verified"] is False
    assert summary["runtime"]["per_node_fallback_status"] == "unknown"
    assert summary["protocol"]["graph_passes_per_sample"] == {
        "prefill": 1,
        "cached_decode": 31,
        "total": 32,
    }
    assert summary["protocol"]["graph_passes_per_condition_per_run"] == {
        "prefill": 30,
        "cached_decode": 930,
        "total": 960,
    }
    assert summary["protocol"]["measured_graph_passes_per_run"] == {
        "prefill": 240,
        "cached_decode": 7440,
        "total": 7680,
    }
    assert summary["protocol"]["measured_graph_passes_all_runs"] == {
        "prefill": 720,
        "cached_decode": 22320,
        "total": 23040,
    }
    assert summary["protocol"]["cache_residency_requested_and_reported"] == "gpu-buffer"
    assert summary["protocol"]["next_token_residency"] == "cpu"
    assert summary["protocol"]["cache_data_read_to_javascript"] is False
    assert summary["protocol"]["all_decode_passes_report_direct_binding_without_readback"] is True
    assert summary["protocol"]["all_superseded_and_final_cache_tensors_disposed"] is True

    for artifact in summary["raw_artifacts"]:
        path = ROOT / artifact["tracked_path"]
        raw = path.read_bytes()
        assert len(raw) == artifact["bytes"]
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]

        run = json.loads(raw)
        assert run["schema_version"] == 1
        assert run["status"] == artifact["status"] == "complete"
        assert run["benchmark"] == "localagent_matched_cached_autoregressive_decode_latency"
        assert run["latency_only"] is True
        assert run["untrained_random_weights"] is True
        assert run["capability_artifact"] is False
        assert run["quality_evaluation"] is False
        assert run["warning"] == (
            "UNTRAINED RANDOM WEIGHTS — LATENCY ONLY; NOT A CAPABILITY OR QUALITY ARTIFACT."
        )
        assert run["failures"] == []
        assert run["errors"] == []

        metadata = run["metadata"]
        _assert_provider_evidence(metadata["provider"])
        assert metadata["ort_version_pin"] == metadata["ort_version_reported"] == "1.27.0"
        assert metadata["ort_version_verified"] is True
        assert metadata["ort_version_verification_status"] == "matches_script_pin"
        assert metadata["protocol_version"] == "cached-decode-latency-0.1"
        assert metadata["context_lengths"] == list(CONTEXT_LENGTHS)
        assert metadata["output_tokens_per_condition"] == 32
        assert metadata["warmups_per_condition"] == 3
        assert metadata["measured_repetitions_per_condition"] == 30
        assert metadata["concurrency"] == 1
        assert metadata["run_once_reload_required"] is True
        assert metadata["tab_visibility_required"] is True
        assert metadata["estimand"] == "prefill_and_iterative_cache_bearing_graph_latency"
        assert metadata["graph_pass_contract"] == {
            "prefill_per_condition": 1,
            "decode_per_condition": 31,
            "total_per_condition": 32,
            "first_token_source": "prefill.next_token",
            "remaining_token_source": "decode.next_token",
        }
        assert metadata["cache_contract"] == {
            "enabled": True,
            "webgpu_cache_residency": "gpu-buffer",
            "wasm_cache_residency": "cpu",
            "next_token_residency": "cpu",
            "update_strategy": (
                "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization"
            ),
            "cache_data_read_to_javascript": False,
            "superseded_and_final_cache_disposal_attempted": True,
        }
        _assert_session_evidence(run)

        measured = run["records"]
        warmups = run["warmup_records"]
        assert len(measured) == artifact["attempted"] == 240
        assert len(warmups) == 24
        assert Counter((row["arm_id"], row["input_tokens"]) for row in measured) == Counter(
            {(arm, context): 30 for arm in ARMS for context in CONTEXT_LENGTHS}
        )
        assert Counter((row["arm_id"], row["input_tokens"]) for row in warmups) == Counter(
            {(arm, context): 3 for arm in ARMS for context in CONTEXT_LENGTHS}
        )
        for record in measured:
            _assert_record_labels(record, "measured")
            _assert_pass_and_cache_evidence(record)
        for record in warmups:
            _assert_record_labels(record, "warmup")
            _assert_pass_and_cache_evidence(record)

        completed = [record for record in measured if record["run_ok"]]
        assert len(completed) == artifact["completed"] == 240
        assert len(measured) - len(completed) == artifact["failed"] == 0
        assert run["summary"]["attempted"] == 240
        assert run["summary"]["completed"] == 240
        assert run["summary"]["failed"] == 0

        conditions: dict[tuple[str, int, str], dict[str, float]] = {}
        for arm in ARMS:
            for input_tokens in CONTEXT_LENGTHS:
                records = [
                    record
                    for record in completed
                    if record["arm_id"] == arm and record["input_tokens"] == input_tokens
                ]
                assert len(records) == 30
                for metric, raw_path in RAW_METRICS.items():
                    values = [_metric_value(record, raw_path) for record in records]
                    conditions[(arm, input_tokens, metric)] = {
                        "p50": _percentile(values, 0.50),
                        "p95": _percentile(values, 0.95),
                    }
        per_run.append(conditions)

    assert summary["aggregate"]["page_session_runs"] == len(per_run) == 3
    assert summary["aggregate"]["measured_attempted"] == 720
    assert summary["aggregate"]["measured_completed"] == 720
    assert summary["aggregate"]["measured_failed"] == 0
    assert summary["aggregate"]["warmups_completed"] == 72

    for condition in summary["conditions"]:
        input_tokens = condition["input_tokens"]
        for metric in RAW_METRICS:
            expected_metric = condition["metrics"][metric]
            observed_by_arm: dict[str, list[dict[str, float]]] = {}
            for arm in ARMS:
                observed_by_arm[arm] = [
                    run[(arm, input_tokens, metric)] for run in per_run
                ]
                summary_arm = "attention" if arm == "webgpu-35m-attn" else "hybrid"
                for percentile in ("p50", "p95"):
                    observed = [
                        run_percentiles[percentile]
                        for run_percentiles in observed_by_arm[arm]
                    ]
                    assert expected_metric[summary_arm][f"{percentile}_median"] == pytest.approx(
                        median(observed), abs=5e-7
                    )
                    assert expected_metric[summary_arm][f"{percentile}_range"] == pytest.approx(
                        [min(observed), max(observed)], abs=5e-7
                    )

            for percentile in ("p50", "p95"):
                ratios = [
                    observed_by_arm["webgpu-35m-hybrid"][index][percentile]
                    / observed_by_arm["webgpu-35m-attn"][index][percentile]
                    for index in range(len(per_run))
                ]
                ratio_summary = expected_metric["hybrid_over_attention"]
                assert ratio_summary[f"{percentile}_median_of_run_ratios"] == pytest.approx(
                    median(ratios), abs=5e-7
                )
                assert ratio_summary[f"{percentile}_range_of_run_ratios"] == pytest.approx(
                    [min(ratios), max(ratios)], abs=5e-7
                )
