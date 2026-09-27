from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/paper/results/m5-webgpu-backbone-20260728.summary.json"


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def test_tracked_webgpu_raw_artifacts_reproduce_the_summary() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    per_run: list[dict[tuple[str, int], dict[str, float]]] = []

    for artifact in summary["raw_artifacts"]:
        path = ROOT / artifact["tracked_path"]
        raw = path.read_bytes()
        assert len(raw) == artifact["bytes"]
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]

        run = json.loads(raw)
        measured = [record for record in run["records"] if record["phase"] == "measured"]
        completed = [record for record in measured if record["run_ok"]]
        assert run["status"] == artifact["status"] == "complete"
        assert len(measured) == 240
        assert len(completed) == artifact["completed"] == 240
        assert len(measured) - len(completed) == artifact["failed"] == 0

        conditions: dict[tuple[str, int], dict[str, float]] = {}
        for arm in ("webgpu-35m-attn", "webgpu-35m-hybrid"):
            for input_tokens in (128, 512, 1024, 1536):
                values = [
                    float(record["inference_ms"])
                    for record in completed
                    if record["arm_id"] == arm and record["input_tokens"] == input_tokens
                ]
                assert len(values) == 30
                conditions[(arm, input_tokens)] = {
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                }
        per_run.append(conditions)

    assert summary["aggregate"]["runs"] == len(per_run) == 3
    assert summary["aggregate"]["attempted"] == 720
    assert summary["aggregate"]["completed"] == 720
    assert summary["aggregate"]["failed"] == 0

    for expected in summary["conditions"]:
        tokens = expected["input_tokens"]
        attention = [run[("webgpu-35m-attn", tokens)] for run in per_run]
        hybrid = [run[("webgpu-35m-hybrid", tokens)] for run in per_run]

        for arm_name, values in (("attention", attention), ("hybrid", hybrid)):
            for percentile in ("p50", "p95"):
                observed = [run[percentile] for run in values]
                assert expected[f"{arm_name}_{percentile}_median_ms"] == pytest.approx(
                    median(observed)
                )
                assert expected[f"{arm_name}_{percentile}_range_ms"] == pytest.approx(
                    [min(observed), max(observed)]
                )

        for percentile in ("p50", "p95"):
            ratios = [
                attention[index][percentile] / hybrid[index][percentile]
                for index in range(len(per_run))
            ]
            key = f"attention_over_hybrid_{percentile}_median_of_run_ratios"
            assert expected[key] == pytest.approx(median(ratios), abs=5e-5)
