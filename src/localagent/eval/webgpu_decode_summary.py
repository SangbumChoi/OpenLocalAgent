"""Reproducible aggregation for trained cached-decode browser measurements."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ARMS = ("webgpu-10m-attn", "webgpu-10m-hybrid")
ARM_LABELS = {
    "webgpu-10m-attn": "attention",
    "webgpu-10m-hybrid": "hybrid",
}
METRICS = {
    "ttft_ms": "ttft_ms",
    "tpot_ms": "tpot_ms",
    "wall_decode_tokens_per_second": "decode_tokens_per_second",
    "model_only_decode_tokens_per_second": "model_decode_tokens_per_second",
    "final_logical_cache_bytes": "cache.final_logical_bytes",
}


def _identity(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return loaded


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
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


def _round(value: float) -> float:
    return round(value, 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_semantics(run: dict[str, Any], path: Path) -> None:
    label = f"{path}:"
    _require(run.get("schema_version") == 1, f"{label} unsupported schema")
    _require(run.get("status") == "complete", f"{label} benchmark is not complete")
    _require(run.get("trained_weights") is True, f"{label} weights are not trained")
    _require(
        run.get("untrained_random_weights") is False,
        f"{label} random-weight label is inconsistent",
    )
    _require(run.get("latency_only") is True, f"{label} result is not latency-only")
    _require(
        run.get("capability_artifact") is False,
        f"{label} result cannot be a capability artifact",
    )
    _require(
        run.get("action_capability_claimed") is False,
        f"{label} action capability must not be claimed",
    )
    _require(
        run.get("action_capability_evaluation") is False,
        f"{label} action capability was not evaluated",
    )
    _require(
        run.get("quality_evaluation") is False,
        f"{label} browser payload cannot contain quality evaluation",
    )
    _require(
        run.get("quality_scored_separately") is True,
        f"{label} trained weights require separate quality evidence",
    )
    _require(not run.get("failures"), f"{label} benchmark contains failures")
    _require(not run.get("errors"), f"{label} benchmark contains errors")

    metadata = run["metadata"]
    _require(metadata["artifact_mode"] == "trained", f"{label} wrong artifact mode")
    _require(metadata["manifest"]["trained"] is True, f"{label} manifest is not trained")
    _require(
        metadata["manifest"]["latency_only"] is False,
        f"{label} trained manifest must defer rather than forbid separate quality evidence",
    )
    _require(
        metadata["manifest"]["capability_artifact"] is False,
        f"{label} manifest cannot claim capability",
    )
    _require(metadata["provider"]["provider_actual"] == "webgpu", f"{label} not WebGPU")
    _require(
        metadata["provider"]["execution_provider_list"] == ["webgpu"],
        f"{label} provider request is not exact-one WebGPU",
    )
    _require(
        metadata["provider"]["whole_session_provider_retry"] is False,
        f"{label} provider fallback/retry occurred",
    )

    for collection in ("records", "warmup_records", "session_records"):
        for record in run[collection]:
            _require(record["trained_weights"] is True, f"{label} mixed trained labels")
            _require(
                record["untrained_random_weights"] is False,
                f"{label} mixed random-weight labels",
            )
            _require(record["latency_only"] is True, f"{label} mixed latency labels")
            _require(
                record["capability_artifact"] is False,
                f"{label} mixed capability labels",
            )
            _require(
                record["quality_evaluation"] is False,
                f"{label} mixed quality labels",
            )
            _require(
                record["quality_scored_separately"] is True,
                f"{label} missing separate-quality label",
            )
            _require(
                record["artifact_manifest_latency_only"] is False,
                f"{label} trained manifest was mislabeled latency-only",
            )
            if collection in ("records", "warmup_records"):
                _require(record["run_ok"] is True, f"{label} unsuccessful timing record")
                _require(record["error"] is None, f"{label} timing record contains an error")
            else:
                _require(record["error"] is None, f"{label} session creation contains an error")

    for arm in metadata["arms"]:
        _require(arm["trained_weights"] is True, f"{label} arm is not trained")
        _require(
            arm["artifact_manifest_latency_only"] is False,
            f"{label} arm manifest semantics are inconsistent",
        )
        _require(
            arm["provenance"]["trained"] is True,
            f"{label} arm provenance is not trained",
        )
        _require(
            arm["provenance"]["latency_only"] is False,
            f"{label} trained provenance must permit separate quality evidence",
        )
        _require(
            arm["provenance"]["capability_artifact"] is False,
            f"{label} arm provenance cannot claim capability",
        )


def _common_contract(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]["metadata"]
    fields = (
        "protocol_version",
        "manifest_sha256",
        "context_lengths",
        "output_tokens_per_condition",
        "warmups_per_condition",
        "measured_repetitions_per_condition",
        "case_order_seed",
        "session_order_seed",
        "input_fixture_contract",
        "input_semantics",
        "estimand",
        "ort_version_pin",
        "ort_version_reported",
        "ort_version_verified",
        "user_agent",
        "device_memory_gb",
        "hardware_concurrency",
        "concurrency",
        "run_once_reload_required",
        "ttft_boundary",
        "tpot_boundary",
        "decode_inference_ms_boundary",
        "manifest",
        "arms",
        "tokenizer_asset",
        "provider",
        "cache_contract",
        "graph_pass_contract",
    )
    for run in runs[1:]:
        metadata = run["metadata"]
        for field in fields:
            _require(metadata[field] == first[field], f"run metadata differs for {field}")
    return first


def _per_run_percentiles(
    runs: list[dict[str, Any]],
    contexts: tuple[int, ...],
    measured_repetitions: int,
) -> list[dict[tuple[str, int, str], dict[str, float]]]:
    per_run: list[dict[tuple[str, int, str], dict[str, float]]] = []
    expected_counts = Counter(
        {
            (arm, context): measured_repetitions
            for arm in ARMS
            for context in contexts
        }
    )
    for run in runs:
        records = run["records"]
        _require(
            Counter((row["arm_id"], row["input_tokens"]) for row in records)
            == expected_counts,
            "measured condition counts differ from the repetition contract",
        )
        conditions: dict[tuple[str, int, str], dict[str, float]] = {}
        for arm in ARMS:
            for context in contexts:
                matching = [
                    row
                    for row in records
                    if row["arm_id"] == arm and row["input_tokens"] == context
                ]
                for metric, raw_path in METRICS.items():
                    values = [_metric_value(record, raw_path) for record in matching]
                    conditions[(arm, context, metric)] = {
                        "p50": _percentile(values, 0.50),
                        "p95": _percentile(values, 0.95),
                    }
        per_run.append(conditions)
    return per_run


def _condition_summaries(
    per_run: list[dict[tuple[str, int, str], dict[str, float]]],
    contexts: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for context in contexts:
        metric_summaries: dict[str, Any] = {}
        for metric in METRICS:
            by_arm: dict[str, list[dict[str, float]]] = {}
            summary: dict[str, Any] = {}
            for arm in ARMS:
                by_arm[arm] = [run[(arm, context, metric)] for run in per_run]
                arm_summary: dict[str, Any] = {}
                for percentile in ("p50", "p95"):
                    values = [row[percentile] for row in by_arm[arm]]
                    arm_summary[f"{percentile}_median"] = _round(median(values))
                    arm_summary[f"{percentile}_range"] = [
                        _round(min(values)),
                        _round(max(values)),
                    ]
                    arm_summary[f"{percentile}_by_run"] = [_round(value) for value in values]
                summary[ARM_LABELS[arm]] = arm_summary

            ratio_summary: dict[str, Any] = {}
            for percentile in ("p50", "p95"):
                ratios = [
                    by_arm["webgpu-10m-hybrid"][index][percentile]
                    / by_arm["webgpu-10m-attn"][index][percentile]
                    for index in range(len(per_run))
                ]
                ratio_summary[f"{percentile}_median_of_run_ratios"] = _round(
                    median(ratios)
                )
                ratio_summary[f"{percentile}_range_of_run_ratios"] = [
                    _round(min(ratios)),
                    _round(max(ratios)),
                ]
            summary["hybrid_over_attention"] = ratio_summary
            metric_summaries[metric] = summary
        output.append({"input_tokens": context, "metrics": metric_summaries})
    return output


def _arm_summary(arm: dict[str, Any]) -> dict[str, Any]:
    provenance = arm["provenance"]
    parity: dict[str, Any] = {}
    for precision, result in provenance["parity"]["results"].items():
        parity[precision] = {
            "provider": result["provider"],
            "passed": result["passed"],
            "greedy_next_token_exact": result["greedy_next_token_exact"],
            "decode_steps": result["decode_steps"],
            "cache_atol": result["cache_atol"],
            "max_cache_abs_diff": result["max_cache_abs_diff"],
        }
    return {
        "arm_id": arm["id"],
        "pair_role": arm["pair_role"],
        "model_parameters": arm["model_parameters"],
        "config": arm["config"],
        "config_source_sha256": arm["config_source_sha256"],
        "config_canonical_sha256": arm["config_canonical_sha256"],
        "checkpoint": {
            "path": provenance["weights"]["checkpoint"],
            "bytes": provenance["weights"]["checkpoint_bytes"],
            "sha256": provenance["weights"]["checkpoint_sha256"],
            "stage": provenance["weights"]["checkpoint_stage"],
            "stored_step_index": provenance["weights"]["checkpoint_step"],
            "completed_updates": provenance["training_steps"],
            "input_tokens_seen": provenance["input_tokens_seen"],
            "loss_tokens_seen": provenance["tokens_seen"],
            "state_dict_sha256": provenance["weights"]["state_dict_sha256"],
        },
        "tokenizer": provenance["tokenizer"],
        "provenance_sha256": arm["provenance_sha256"],
        "graphs": {
            "prefill": {
                "bytes": arm["prefill_bytes"],
                "sha256": arm["prefill_sha256"],
            },
            "decode": {
                "bytes": arm["decode_bytes"],
                "sha256": arm["decode_sha256"],
            },
        },
        "cache_slot_count": arm["cache_slot_count"],
        "cache_tensor_count": arm["cache_tensor_count"],
        "parity": parity,
    }


def _latency_gate(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    by_context: list[dict[str, Any]] = []
    for condition in conditions:
        metrics = condition["metrics"]
        row: dict[str, Any] = {"input_tokens": condition["input_tokens"]}
        for arm in ("attention", "hybrid"):
            wall = metrics["wall_decode_tokens_per_second"][arm]
            tpot = metrics["tpot_ms"][arm]
            median_p50_wall_pass = wall["p50_median"] >= 100.0
            median_p95_tpot_pass = tpot["p95_median"] <= 10.0
            every_run_p50_wall_pass = all(value >= 100.0 for value in wall["p50_by_run"])
            every_run_p95_tpot_pass = all(value <= 10.0 for value in tpot["p95_by_run"])
            row[arm] = {
                "p50_wall_decode_tokens_per_second": wall["p50_median"],
                "p95_tpot_ms": tpot["p95_median"],
                "median_of_run_percentiles": {
                    "p50_wall_at_least_100": median_p50_wall_pass,
                    "p95_tpot_at_most_10_ms": median_p95_tpot_pass,
                    "joint_pass": median_p50_wall_pass and median_p95_tpot_pass,
                },
                "every_page_run": {
                    "p50_wall_at_least_100": every_run_p50_wall_pass,
                    "p95_tpot_at_most_10_ms": every_run_p95_tpot_pass,
                    "joint_pass": every_run_p50_wall_pass and every_run_p95_tpot_pass,
                },
            }
        by_context.append(row)

    return {
        "thresholds": {
            "p50_wall_decode_tokens_per_second_minimum": 100.0,
            "p95_tpot_ms_maximum": 10.0,
        },
        "decision_statistic": "median of three within-run percentiles",
        "stability_statistic": "threshold must also hold in every page/session run",
        "scope": "engineering latency gate only; never a quality or capability gate",
        "by_context": by_context,
        "all_contexts": {
            arm: {
                "median_of_run_percentiles_pass": all(
                    row[arm]["median_of_run_percentiles"]["joint_pass"]
                    for row in by_context
                ),
                "every_page_run_pass": all(
                    row[arm]["every_page_run"]["joint_pass"] for row in by_context
                ),
                "p50_wall_at_least_100_every_page_run": all(
                    row[arm]["every_page_run"]["p50_wall_at_least_100"]
                    for row in by_context
                ),
                "p95_tpot_at_most_10_ms_median": all(
                    row[arm]["median_of_run_percentiles"]["p95_tpot_at_most_10_ms"]
                    for row in by_context
                ),
            }
            for arm in ("attention", "hybrid")
        },
    }


def build_trained_decode_summary(
    raw_paths: list[Path],
    *,
    repository_root: Path,
    quality_summary_path: Path,
    paired_comparison_path: Path,
) -> dict[str, Any]:
    """Build a deterministic summary from independent trained browser result payloads."""
    _require(len(raw_paths) >= 2, "at least two independent page/session runs are required")
    runs = [_load_json(path) for path in raw_paths]
    for path, run in zip(raw_paths, runs, strict=True):
        _validate_semantics(run, path)

    common = _common_contract(runs)
    contexts = tuple(int(value) for value in common["context_lengths"])
    per_run = _per_run_percentiles(
        runs,
        contexts,
        int(common["measured_repetitions_per_condition"]),
    )
    conditions = _condition_summaries(per_run, contexts)
    manifest = common["manifest"]
    arms = {arm["id"]: arm for arm in common["arms"]}

    quality_summary = _load_json(quality_summary_path)
    paired_comparison = _load_json(paired_comparison_path)
    _require(
        quality_summary["arms"]["attention"]["checkpoint"]["sha256"]
        == arms["webgpu-10m-attn"]["checkpoint_sha256"],
        "attention quality and latency checkpoint hashes differ",
    )
    _require(
        quality_summary["arms"]["hybrid"]["checkpoint"]["sha256"]
        == arms["webgpu-10m-hybrid"]["checkpoint_sha256"],
        "hybrid quality and latency checkpoint hashes differ",
    )

    raw_artifacts: list[dict[str, Any]] = []
    for index, (path, run) in enumerate(zip(raw_paths, runs, strict=True), start=1):
        identity = _identity(path, repository_root)
        raw_artifacts.append(
            {
                "run": index,
                "created_at": run["created_at"],
                "tracked_path": identity["path"],
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
                "status": run["status"],
                "attempted": len(run["records"]),
                "completed": sum(record["run_ok"] for record in run["records"]),
                "failed": sum(not record["run_ok"] for record in run["records"]),
                "warmups_completed": sum(
                    record["run_ok"] for record in run["warmup_records"]
                ),
            }
        )

    quality_identity = _identity(quality_summary_path, repository_root)
    comparison_identity = _identity(paired_comparison_path, repository_root)
    graph_passes_per_sample = common["graph_pass_contract"]
    conditions_per_run = len(ARMS) * len(contexts)
    measured_per_condition = common["measured_repetitions_per_condition"]
    run_count = len(runs)

    return {
        "schema_version": 1,
        "artifact_type": "trained_checkpoint_cached_autoregressive_decode_latency_summary",
        "trained": True,
        "pretrain_only": True,
        "untrained_random_weights": False,
        "latency_only": True,
        "capability_artifact": False,
        "action_capability_claimed": False,
        "action_capability_evaluation": False,
        "quality_evaluation_in_browser_payload": False,
        "quality_scored_separately": True,
        "quality_claims": [],
        "first_run_result_created_at": raw_artifacts[0]["created_at"],
        "last_run_result_created_at": raw_artifacts[-1]["created_at"],
        "model_pair": {
            "controlled_fields": manifest["controlled_fields"],
            "intentional_differences": manifest["intentional_differences"],
            "relative_parameter_delta": manifest["match"]["relative_parameter_delta"],
            "tokenizer": manifest["tokenizer"],
            "attention": _arm_summary(arms["webgpu-10m-attn"]),
            "hybrid": _arm_summary(arms["webgpu-10m-hybrid"]),
        },
        "runtime": {
            "onnxruntime_web_pin": common["ort_version_pin"],
            "onnxruntime_web_reported": common["ort_version_reported"],
            "onnxruntime_version_verified": common["ort_version_verified"],
            "execution_provider_request": common["provider"]["execution_provider_list"],
            "execution_provider_actual": common["provider"]["provider_actual"],
            "provider_actual_scope": common["provider"]["provider_actual_scope"],
            "whole_session_provider_retry": common["provider"][
                "whole_session_provider_retry"
            ],
            "ort_webgpu_device_available": common["provider"]["ort_webgpu"][
                "ort_device_available"
            ],
            "adapter_introspection_available": common["provider"]["ort_webgpu"][
                "ort_adapter_available"
            ],
            "per_node_placement_verified": common["provider"][
                "per_node_placement_verified"
            ],
            "per_node_fallback_status": common["provider"]["per_node_fallback_status"],
            "browser_user_agent": common["user_agent"],
            "device_memory_gb_reported": common["device_memory_gb"],
            "hardware_concurrency_reported": common["hardware_concurrency"],
            "concurrency": common["concurrency"],
        },
        "protocol": {
            "protocol_version": common["protocol_version"],
            "estimand": common["estimand"],
            "manifest_sha256": common["manifest_sha256"],
            "context_lengths": list(contexts),
            "deterministic_input_contract": common["input_fixture_contract"],
            "input_semantics": common["input_semantics"],
            "output_tokens_per_sample": common["output_tokens_per_condition"],
            "warmups_per_condition": common["warmups_per_condition"],
            "measured_repetitions_per_condition": measured_per_condition,
            "run_once_reload_required": common["run_once_reload_required"],
            "graph_passes_per_sample": {
                "prefill": graph_passes_per_sample["prefill_per_condition"],
                "cached_decode": graph_passes_per_sample["decode_per_condition"],
                "total": graph_passes_per_sample["total_per_condition"],
            },
            "measured_graph_passes_all_runs": {
                "prefill": run_count * conditions_per_run * measured_per_condition,
                "cached_decode": (
                    run_count
                    * conditions_per_run
                    * measured_per_condition
                    * graph_passes_per_sample["decode_per_condition"]
                ),
                "total": (
                    run_count
                    * conditions_per_run
                    * measured_per_condition
                    * graph_passes_per_sample["total_per_condition"]
                ),
            },
            "first_token_source": graph_passes_per_sample["first_token_source"],
            "remaining_token_source": graph_passes_per_sample["remaining_token_source"],
            "ttft_boundary": common["ttft_boundary"],
            "tpot_boundary": common["tpot_boundary"],
            "model_only_boundary": common["decode_inference_ms_boundary"],
            "precision": "fp16",
            "cache_residency_requested_and_reported": "gpu-buffer",
            "next_token_residency": "cpu",
            "cache_update_strategy": common["cache_contract"]["update_strategy"],
            "cache_data_read_to_javascript": common["cache_contract"][
                "cache_data_read_to_javascript"
            ],
            "all_decode_passes_report_direct_binding_without_readback": all(
                decode_pass["cache_bound_directly_without_readback"]
                for run in runs
                for record in (*run["warmup_records"], *run["records"])
                for decode_pass in record["decode_pass_records"]
            ),
            "all_superseded_and_final_cache_tensors_disposed": all(
                record["allocation_disposal"]["cache_dispose_failed"] == 0
                and record["allocation_disposal"]["cache_dispose_api_unavailable"] == 0
                and record["allocation_disposal"]["cache_dispose_succeeded"]
                == record["allocation_disposal"]["cache_tensors_allocated"]
                for run in runs
                for record in (*run["warmup_records"], *run["records"])
            ),
            "all_fetched_model_artifact_hashes_and_sizes_verified_before_use": all(
                record["verification_before_parse_or_ort"]
                and (
                    (
                        record["artifact_kind"] == "benchmark_manifest"
                        and record["hash_verification_status"]
                        == "unknown_no_external_expected_digest"
                    )
                    or (
                        record["artifact_kind"] != "benchmark_manifest"
                        and record["hash_verified"] is True
                        and record["bytes_verified"] is True
                    )
                )
                for run in runs
                for record in run["artifact_verification_records"]
            ),
        },
        "raw_artifacts": raw_artifacts,
        "aggregate": {
            "page_session_runs": run_count,
            "conditions_per_run": conditions_per_run,
            "measured_repetitions_per_condition_per_run": measured_per_condition,
            "measured_attempted": sum(row["attempted"] for row in raw_artifacts),
            "measured_completed": sum(row["completed"] for row in raw_artifacts),
            "measured_failed": sum(row["failed"] for row in raw_artifacts),
            "warmups_completed": sum(row["warmups_completed"] for row in raw_artifacts),
            "pooling": "none",
            "reported_point_estimate": "median of the within-run percentiles",
            "reported_range": "minimum and maximum of the within-run percentiles",
            "paired_ratio_estimator": (
                "median and full range of within-run hybrid_over_attention ratios"
            ),
        },
        "conditions": conditions,
        "latency_gate": _latency_gate(conditions),
        "quality_evidence": {
            "measurement_relation": (
                "same checkpoint hashes, separately measured held-out next-token quality"
            ),
            "latency_payload_contains_quality_metrics": False,
            "scope": "pretraining language-model quality only; no agent/action capability",
            "summary_artifact": quality_identity,
            "paired_document_bootstrap_artifact": {
                **comparison_identity,
                "canonical_comparison_sha256": paired_comparison["comparison_sha256"],
            },
            "bootstrap": paired_comparison["bootstrap"],
            "heldout_scorecards": {
                "attention": quality_summary["arms"]["attention"]["scorecard"],
                "hybrid": quality_summary["arms"]["hybrid"]["scorecard"],
            },
            "paired_document_differences_attention_minus_hybrid": {
                "overall": paired_comparison["overall"][
                    "difference_attention_minus_hybrid"
                ],
                "general": paired_comparison["groups"]["general"][
                    "difference_attention_minus_hybrid"
                ],
                "code": paired_comparison["groups"]["code"][
                    "difference_attention_minus_hybrid"
                ],
            },
        },
        "interpretation": {
            "supported": (
                "For these exact one-seed pretraining checkpoints on this M5/Chrome/ORT run, "
                "the hybrid clears 100 wall decode tok/s at every tested context in every "
                "page run and has better separate held-out CE/BPB/accuracy."
            ),
            "latency_gate_result": (
                "partial: the 100 tok/s p50 component clears, but the joint gate fails "
                "because median-of-run p95 TPOT exceeds 10 ms at three contexts"
            ),
            "not_supported": [
                "agent, tool-use, structured-output, or browser-task capability",
                "multi-seed architecture selection",
                "cross-device or cross-browser generalization",
                "a full p50-throughput plus p95-tail-latency acceptance pass",
            ],
        },
        "limitations": [
            (
                "Both checkpoints are pretrain-only proxy models trained for approximately "
                "one token per parameter; no midtrain, SFT, or RL is represented."
            ),
            (
                "Quality and latency are linked by checkpoint hash but measured in separate "
                "evaluations; the browser payload itself contains no quality metric."
            ),
            (
                "Document-bootstrap intervals condition on one architecture seed and one "
                "held-out set; they are not multi-seed architecture uncertainty."
            ),
            (
                "Each point is the median of separately summarized page/session runs; "
                "samples are not pooled and the range is not a confidence interval."
            ),
            (
                "Adapter identity and per-node placement are unavailable even though an "
                "exact-one WebGPU provider request created every session."
            ),
            (
                "The result covers one Apple M5, Chrome 150, and ONNX Runtime Web 1.27.0; "
                "other browsers and GPUs remain unmeasured."
            ),
        ],
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON result artifact."""
    path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
