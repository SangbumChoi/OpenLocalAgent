from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from localagent.eval.pretrain_compare import compare_pretrain_sidecars
from localagent.eval.pretrain_seed_aggregate import (
    SeedComparisonSpec,
    aggregate_pretrain_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "paper" / "results"
RAW = RESULTS / "raw" / "pretrain-proxy-seeds2027-2029"
SUMMARY = RESULTS / "webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json"
SEEDS = (2027, 2028, 2029)
EXPECTED_SUMMARY_FILE_SHA256 = (
    "feb6e2c601f7692ebd25f53b878987b0916576a62db90c7ca878f4b467000669"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _specification(seed: int) -> SeedComparisonSpec:
    return SeedComparisonSpec(
        seed=seed,
        attention_config=(
            Path("configs")
            / "train"
            / f"pretrain-webgpu-proxy-1tpp-attn-seed{seed}.yaml"
        ),
        hybrid_config=(
            Path("configs")
            / "train"
            / f"pretrain-webgpu-proxy-1tpp-hybrid-seed{seed}.yaml"
        ),
        comparison=(
            Path("docs")
            / "paper"
            / "results"
            / "raw"
            / "pretrain-proxy-seeds2027-2029"
            / f"seed{seed}.paired.json"
        ),
    )


def test_tracked_multiseed_proxy_artifacts_recompute_exactly() -> None:
    for seed in SEEDS:
        expected_comparison = json.loads(
            (RAW / f"seed{seed}.paired.json").read_text(encoding="utf-8")
        )
        recomputed_comparison = compare_pretrain_sidecars(
            (
                Path("docs")
                / "paper"
                / "results"
                / "raw"
                / "pretrain-proxy-seeds2027-2029"
                / f"seed{seed}-attn.documents.jsonl"
            ),
            (
                Path("docs")
                / "paper"
                / "results"
                / "raw"
                / "pretrain-proxy-seeds2027-2029"
                / f"seed{seed}-hybrid.documents.jsonl"
            ),
            seed=2026,
            resamples=10_000,
            confidence=0.95,
        )
        assert recomputed_comparison == expected_comparison

    expected_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    recomputed_summary = aggregate_pretrain_seeds(
        [_specification(seed) for seed in SEEDS]
    )
    assert recomputed_summary == expected_summary
    assert _sha256_file(SUMMARY) == EXPECTED_SUMMARY_FILE_SHA256


def test_multiseed_proxy_primary_result_is_reported_at_seed_level() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["design"]["training_seeds"] == list(SEEDS)
    assert summary["design"]["unit_of_replication"] == "training_seed"
    assert summary["design"]["primary_metric"] == "overall.bits_per_byte"
    assert summary["evaluation"]["device"] == "cpu"
    primary = summary["overall"]["metrics"]["bits_per_byte"]
    assert [row["estimate"] for row in primary["seed_estimates"]] == pytest.approx(
        [0.07083587527346794, 0.0735279993681252, 0.07448855087281897]
    )
    difference = primary["difference_attention_minus_hybrid"]
    assert difference["mean"] == pytest.approx(0.07295080850480402)
    assert difference["hybrid_favoring_seeds"] == 3
    assert difference["attention_favoring_seeds"] == 0
    assert difference["student_t_95_interval"]["lower"] == pytest.approx(
        0.06824707441091234
    )
    assert difference["student_t_95_interval"]["upper"] == pytest.approx(
        0.07765454259869571
    )
    assert difference["exact_sign_test"]["majority_one_sided_p_value"] == 0.125
    assert difference["exact_sign_test"]["two_sided_p_value"] == 0.25

    for metric_name in (
        "bits_per_byte",
        "cross_entropy_nats_per_token",
        "top1_accuracy",
    ):
        metric = summary["overall"]["metrics"][metric_name]
        assert metric["difference_attention_minus_hybrid"]["hybrid_favoring_seeds"] == 3
        assert len(metric["seed_estimates"]) == 3
