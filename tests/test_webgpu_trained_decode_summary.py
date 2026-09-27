from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pytest

from localagent.eval.webgpu_decode_summary import build_trained_decode_summary


ROOT = Path(__file__).resolve().parents[1]
RAW_PATHS = [
    ROOT
    / (
        "docs/paper/results/raw/"
        f"m5-webgpu-cached-decode-10m-trained-proxy-20260728-run{run}.json"
    )
    for run in (1, 2, 3)
]
SUMMARY_PATH = (
    ROOT
    / "docs/paper/results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json"
)
QUALITY_SUMMARY_PATH = (
    ROOT / "docs/paper/results/webgpu-proxy-1tpp-10m-seed2026.summary.json"
)
COMPARISON_PATH = (
    ROOT / "runs/pretrain-webgpu-proxy-1tpp-paired-comparison-seed2026.json"
)
ARMS = ("webgpu-10m-attn", "webgpu-10m-hybrid")
CONTEXTS = (128, 512, 1024, 1536)
RAW_METRICS = {
    "ttft_ms": "ttft_ms",
    "tpot_ms": "tpot_ms",
    "wall_decode_tokens_per_second": "decode_tokens_per_second",
    "model_only_decode_tokens_per_second": "model_decode_tokens_per_second",
    "final_logical_cache_bytes": "cache.final_logical_bytes",
}
EXPECTED_CHECKPOINTS = {
    "webgpu-10m-attn": (
        "b86929f708b0294ff305fa9ffbfa5059e04a807facfc0c5c55d64c471215f4a9"
    ),
    "webgpu-10m-hybrid": (
        "00dd2cf6651b0a27e18d707d287b464361e4f0636c7c787fafc7570682ab2e6d"
    ),
}
EXPECTED_GRAPHS = {
    "webgpu-10m-attn": {
        "prefill": (
            34_117_852,
            "1e8103fe99e7d9aaac3a6ab5c06f0a5c4768624d9584f86d14e6bd598237888e",
        ),
        "decode": (
            34_109_739,
            "8058b1a31d4a3851ef77cf10079eaf0d95c29cc4e6b8bbb35847e35efb94c012",
        ),
    },
    "webgpu-10m-hybrid": {
        "prefill": (
            34_015_397,
            "d9f47f917f6065701c2c79f2fcc7ae1ba0873dad888c122fc9a1c580960f69cd",
        ),
        "decode": (
            34_009_630,
            "4e8f34aad920997300bd06ec909cb681978b5c12babd50a209b3a29e868afa9c",
        ),
    },
}
TRAINED_SUMMARY_ARTIFACTS = (
    *RAW_PATHS,
    SUMMARY_PATH,
    QUALITY_SUMMARY_PATH,
    COMPARISON_PATH,
)
TRAINED_SUMMARY_BUILDER_INPUTS = (
    *RAW_PATHS,
    QUALITY_SUMMARY_PATH,
    COMPARISON_PATH,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_value(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    return float(value)


def _assert_trained_latency_labels(record: dict[str, Any]) -> None:
    assert record["trained_weights"] is True
    assert record["untrained_random_weights"] is False
    assert record["latency_only"] is True
    assert record["capability_artifact"] is False
    assert record["action_capability_claimed"] is False
    assert record["action_capability_evaluation"] is False
    assert record["quality_evaluation"] is False
    assert record["quality_scored_separately"] is True
    assert record["artifact_manifest_latency_only"] is False
    assert record["benchmark_label"] == (
        "trained weights, latency only; quality scored separately"
    )


@pytest.mark.skipif(
    not all(path.is_file() for path in TRAINED_SUMMARY_ARTIFACTS),
    reason="requires the exact tracked browser payloads and local sealed paired comparison",
)
def test_trained_browser_payloads_reproduce_the_tracked_summary() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    rebuilt = build_trained_decode_summary(
        RAW_PATHS,
        repository_root=ROOT,
        quality_summary_path=QUALITY_SUMMARY_PATH,
        paired_comparison_path=COMPARISON_PATH,
    )
    assert summary == rebuilt

    assert summary["artifact_type"] == (
        "trained_checkpoint_cached_autoregressive_decode_latency_summary"
    )
    assert summary["trained"] is True
    assert summary["pretrain_only"] is True
    assert summary["untrained_random_weights"] is False
    assert summary["latency_only"] is True
    assert summary["capability_artifact"] is False
    assert summary["action_capability_claimed"] is False
    assert summary["action_capability_evaluation"] is False
    assert summary["quality_evaluation_in_browser_payload"] is False
    assert summary["quality_scored_separately"] is True
    assert summary["quality_claims"] == []

    raw_runs: list[dict[str, Any]] = []
    for path, artifact in zip(RAW_PATHS, summary["raw_artifacts"], strict=True):
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha256(path)
        run = json.loads(path.read_text())
        raw_runs.append(run)

        assert run["status"] == artifact["status"] == "complete"
        assert run["summary"]["estimand"] == (
            "trained_weight_cached_autoregressive_graph_latency"
        )
        assert run["summary"]["quality_metrics_included"] is False
        assert run["failures"] == []
        assert run["errors"] == []
        _assert_trained_latency_labels(run)

        metadata = run["metadata"]
        assert metadata["artifact_mode"] == "trained"
        assert metadata["manifest"]["artifact_type"] == (
            "matched_trained_cached_decode_suite"
        )
        assert metadata["manifest"]["trained"] is True
        assert metadata["manifest"]["latency_only"] is False
        assert metadata["manifest"]["capability_artifact"] is False
        assert metadata["manifest"]["quality_claims"] == []
        assert metadata["provider"]["provider_actual"] == "webgpu"
        assert metadata["provider"]["execution_provider_list"] == ["webgpu"]
        assert metadata["provider"]["whole_session_provider_retry"] is False
        assert metadata["provider"]["per_node_placement_verified"] is False
        assert metadata["ort_version_pin"] == metadata["ort_version_reported"] == "1.27.0"
        assert metadata["ort_version_verified"] is True
        assert metadata["context_lengths"] == list(CONTEXTS)
        assert metadata["output_tokens_per_condition"] == 32
        assert metadata["warmups_per_condition"] == 3
        assert metadata["measured_repetitions_per_condition"] == 30
        assert hashlib.sha256(metadata["manifest_raw_text"].encode()).hexdigest() == metadata[
            "manifest_sha256"
        ]

        measured = run["records"]
        warmups = run["warmup_records"]
        assert len(measured) == artifact["attempted"] == artifact["completed"] == 240
        assert artifact["failed"] == 0
        assert len(warmups) == artifact["warmups_completed"] == 24
        assert Counter((row["arm_id"], row["input_tokens"]) for row in measured) == Counter(
            {(arm, context): 30 for arm in ARMS for context in CONTEXTS}
        )
        assert Counter((row["arm_id"], row["input_tokens"]) for row in warmups) == Counter(
            {(arm, context): 3 for arm in ARMS for context in CONTEXTS}
        )
        for record in (*measured, *warmups, *run["session_records"]):
            _assert_trained_latency_labels(record)

        verification = run["artifact_verification_records"]
        assert len(verification) == 9
        for artifact_record in verification:
            assert artifact_record["verification_before_parse_or_ort"] is True
            if artifact_record["artifact_kind"] == "benchmark_manifest":
                assert artifact_record["hash_verified"] is None
                assert artifact_record["hash_verification_status"] == (
                    "unknown_no_external_expected_digest"
                )
            else:
                assert artifact_record["hash_verified"] is True
                assert artifact_record["bytes_verified"] is True

        for record in (*measured, *warmups):
            assert record["run_ok"] is True
            assert {
                key: record["graph_pass_counts"][key]
                for key in ("prefill", "decode", "total")
            } == {
                "prefill": 1,
                "decode": 31,
                "total": 32,
            }
            assert len(record["decode_pass_records"]) == 31
            assert all(
                row["cache_reported_locations"] == ["gpu-buffer"]
                and row["cache_bound_directly_without_readback"] is True
                for row in record["decode_pass_records"]
            )
            disposal = record["allocation_disposal"]
            assert disposal["cache_dispose_failed"] == 0
            assert disposal["cache_dispose_api_unavailable"] == 0
            assert disposal["cache_dispose_succeeded"] == disposal[
                "cache_tensors_allocated"
            ]

        arms = {arm["id"]: arm for arm in metadata["arms"]}
        assert set(arms) == set(ARMS)
        for arm_id, arm in arms.items():
            assert arm["checkpoint_sha256"] == EXPECTED_CHECKPOINTS[arm_id]
            assert arm["tokens_seen"] == 10_551_291
            assert arm["checkpoint_step"] == 321
            assert arm["tokenizer_sha256"] == (
                "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c"
            )
            for graph_kind, (expected_bytes, expected_sha) in EXPECTED_GRAPHS[
                arm_id
            ].items():
                assert arm[f"{graph_kind}_bytes"] == expected_bytes
                assert arm[f"{graph_kind}_sha256"] == expected_sha
            provenance = arm["provenance"]
            assert provenance["trained"] is True
            assert provenance["latency_only"] is False
            assert provenance["capability_artifact"] is False
            assert provenance["input_tokens_seen"] == 10_551_296
            assert provenance["tokens_seen"] == 10_551_291
            assert provenance["weights"]["checkpoint_sha256"] == EXPECTED_CHECKPOINTS[
                arm_id
            ]
            for parity in provenance["parity"]["results"].values():
                assert parity["passed"] is True
                assert parity["greedy_next_token_exact"] is True
                assert parity["max_cache_abs_diff"] <= parity["cache_atol"]

        assert len(run["session_records"]) == 4
        for session in run["session_records"]:
            arm = arms[session["arm_id"]]
            graph_kind = session["graph_kind"]
            assert session["graph_bytes"] == arm[f"{graph_kind}_bytes"]
            assert session["graph_sha256"] == arm[f"{graph_kind}_sha256"]
            assert session["provider_actual"] == "webgpu"
            assert session["execution_provider_list"] == ["webgpu"]
            assert session["cache_residency_requested"] == "gpu-buffer"
            assert session["next_token_residency_requested"] == "cpu"

    assert summary["aggregate"] == {
        "page_session_runs": 3,
        "conditions_per_run": 8,
        "measured_repetitions_per_condition_per_run": 30,
        "measured_attempted": 720,
        "measured_completed": 720,
        "measured_failed": 0,
        "warmups_completed": 72,
        "pooling": "none",
        "reported_point_estimate": "median of the within-run percentiles",
        "reported_range": "minimum and maximum of the within-run percentiles",
        "paired_ratio_estimator": (
            "median and full range of within-run hybrid_over_attention ratios"
        ),
    }
    assert summary["protocol"]["measured_graph_passes_all_runs"] == {
        "prefill": 720,
        "cached_decode": 22_320,
        "total": 23_040,
    }
    assert summary["protocol"][
        "all_decode_passes_report_direct_binding_without_readback"
    ]
    assert summary["protocol"]["all_superseded_and_final_cache_tensors_disposed"]
    assert summary["protocol"][
        "all_fetched_model_artifact_hashes_and_sizes_verified_before_use"
    ]

    conditions = {row["input_tokens"]: row for row in summary["conditions"]}
    for context in CONTEXTS:
        for metric, raw_path in RAW_METRICS.items():
            observed_by_arm: dict[str, dict[str, list[float]]] = {}
            for arm_id, arm_label in zip(ARMS, ("attention", "hybrid"), strict=True):
                records_by_run = [
                    [
                        row
                        for row in run["records"]
                        if row["arm_id"] == arm_id and row["input_tokens"] == context
                    ]
                    for run in raw_runs
                ]
                observed_by_arm[arm_id] = {
                    percentile: [
                        _percentile(
                            [_metric_value(row, raw_path) for row in records],
                            quantile,
                        )
                        for records in records_by_run
                    ]
                    for percentile, quantile in (("p50", 0.50), ("p95", 0.95))
                }
                reported = conditions[context]["metrics"][metric][arm_label]
                for percentile in ("p50", "p95"):
                    values = observed_by_arm[arm_id][percentile]
                    assert reported[f"{percentile}_median"] == pytest.approx(
                        median(values), abs=5e-7
                    )
                    assert reported[f"{percentile}_range"] == pytest.approx(
                        [min(values), max(values)], abs=5e-7
                    )
                    assert reported[f"{percentile}_by_run"] == pytest.approx(
                        values, abs=5e-7
                    )

            ratios = conditions[context]["metrics"][metric][
                "hybrid_over_attention"
            ]
            for percentile in ("p50", "p95"):
                observed_ratios = [
                    hybrid / attention
                    for hybrid, attention in zip(
                        observed_by_arm["webgpu-10m-hybrid"][percentile],
                        observed_by_arm["webgpu-10m-attn"][percentile],
                        strict=True,
                    )
                ]
                assert ratios[
                    f"{percentile}_median_of_run_ratios"
                ] == pytest.approx(median(observed_ratios), abs=5e-7)
                assert ratios[
                    f"{percentile}_range_of_run_ratios"
                ] == pytest.approx(
                    [min(observed_ratios), max(observed_ratios)],
                    abs=5e-7,
                )

    gate = summary["latency_gate"]
    assert gate["thresholds"] == {
        "p50_wall_decode_tokens_per_second_minimum": 100.0,
        "p95_tpot_ms_maximum": 10.0,
    }
    assert gate["all_contexts"]["hybrid"] == {
        "median_of_run_percentiles_pass": False,
        "every_page_run_pass": False,
        "p50_wall_at_least_100_every_page_run": True,
        "p95_tpot_at_most_10_ms_median": False,
    }
    assert gate["all_contexts"]["attention"]["median_of_run_percentiles_pass"] is False
    assert [
        row["hybrid"]["median_of_run_percentiles"]["joint_pass"]
        for row in gate["by_context"]
    ] == [False, True, False, False]

    quality = summary["quality_evidence"]
    assert quality["summary_artifact"]["sha256"] == _sha256(QUALITY_SUMMARY_PATH)
    assert quality["paired_document_bootstrap_artifact"]["sha256"] == _sha256(
        COMPARISON_PATH
    )
    assert quality["paired_document_bootstrap_artifact"][
        "canonical_comparison_sha256"
    ] == "3429a7a99ef233e445b8510288d340155c234e12fa3e5f31f24e12895396a400"
    assert quality["bootstrap"]["resamples"] == 10_000
    assert quality["bootstrap"]["seed"] == 2026
    assert quality["scope"] == (
        "pretraining language-model quality only; no agent/action capability"
    )

    proxy = json.loads(QUALITY_SUMMARY_PATH.read_text())
    bootstrap = proxy["paired_document_bootstrap"]
    assert bootstrap["artifact_bytes"] == COMPARISON_PATH.stat().st_size
    assert bootstrap["artifact_file_sha256"] == _sha256(COMPARISON_PATH)
    assert bootstrap["comparison_canonical_sha256"] == (
        "3429a7a99ef233e445b8510288d340155c234e12fa3e5f31f24e12895396a400"
    )
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["aggregate"]["cross_entropy_nats_per_token"][
        "percentile_ci"
    ] == pytest.approx([0.18345915201354507, 0.2034141987837352])
    assert bootstrap["aggregate"]["bits_per_byte"]["percentile_ci"] == pytest.approx(
        [0.06341848245860873, 0.07072525021190731]
    )
    assert bootstrap["aggregate"]["top1_accuracy"]["percentile_ci"] == pytest.approx(
        [-0.022711271993261593, -0.018343023404995426]
    )
    trained_result = proxy["trained_webgpu_result"]
    assert trained_result["status"] == "collected_partial_gate"
    assert trained_result["p50_wall_at_least_100_every_context_and_page_run"] is True
    assert trained_result["joint_p50_wall_and_p95_tpot_gate_pass"] is False
    assert trained_result["capability_artifact"] is False


@pytest.mark.skipif(
    not all(path.is_file() for path in TRAINED_SUMMARY_BUILDER_INPUTS),
    reason="requires the exact tracked browser payloads and local sealed paired comparison",
)
def test_summary_builder_rejects_a_quality_checkpoint_mismatch(tmp_path: Path) -> None:
    quality = json.loads(QUALITY_SUMMARY_PATH.read_text())
    quality["arms"]["hybrid"]["checkpoint"]["sha256"] = "0" * 64
    mismatched = tmp_path / "quality.json"
    mismatched.write_text(json.dumps(quality))

    with pytest.raises(
        ValueError,
        match="hybrid quality and latency checkpoint hashes differ",
    ):
        build_trained_decode_summary(
            RAW_PATHS,
            repository_root=ROOT,
            quality_summary_path=mismatched,
            paired_comparison_path=COMPARISON_PATH,
        )
