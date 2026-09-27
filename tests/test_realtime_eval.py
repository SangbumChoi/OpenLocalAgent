import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localagent.eval.realtime import (
    autoregressive_ttfa_ms,
    bootstrap_success_at_deadline_ci,
    calibrate_autoregressive_rate_scenarios,
    latency_summary,
    paired_success_at_deadline_delta_ci,
    percentile,
    required_decode_tokens_per_second,
    summarize_action_records,
    summarize_grouped_action_records,
)


def _case_rows(
    case_id: str,
    outcomes: list[bool],
    *,
    ttfa_ms: float = 50.0,
) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "ttfa_ms": ttfa_ms,
            "success": outcome,
            "schema_valid": True,
        }
        for outcome in outcomes
    ]


def test_percentile_uses_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.95) == pytest.approx(38.5)
    assert percentile(values, 1.0) == 40.0


def test_latency_summary_rejects_invalid_samples():
    with pytest.raises(ValueError):
        latency_summary([])
    with pytest.raises(ValueError):
        latency_summary([1.0, -1.0])
    with pytest.raises(ValueError):
        latency_summary([math.inf])
    with pytest.raises(ValueError):
        percentile([1.0, math.nan], 0.5)


def test_autoregressive_action_latency_counts_first_token_in_ttft():
    # First token at 100 ms, then 31 token intervals at 100 tok/s, then a 10 ms parser.
    assert autoregressive_ttfa_ms(
        32, ttft_ms=100.0, decode_tokens_per_second=100.0, postprocess_ms=10.0
    ) == pytest.approx(420.0)


def test_one_token_action_needs_no_iterative_decode_rate():
    assert autoregressive_ttfa_ms(
        1,
        ttft_ms=90.0,
        decode_tokens_per_second=0.0,
        postprocess_ms=10.0,
    ) == 100.0
    with pytest.raises(ValueError, match="positive"):
        autoregressive_ttfa_ms(
            2,
            ttft_ms=90.0,
            decode_tokens_per_second=0.0,
            postprocess_ms=10.0,
        )


def test_required_decode_rate_is_deadline_and_length_dependent():
    assert required_decode_tokens_per_second(
        32, deadline_ms=500.0, ttft_ms=100.0, postprocess_ms=10.0
    ) == pytest.approx(31 / 0.39)
    assert required_decode_tokens_per_second(
        1, deadline_ms=100.0, ttft_ms=90.0, postprocess_ms=10.0
    ) == 0.0
    assert math.isinf(
        required_decode_tokens_per_second(
            2, deadline_ms=100.0, ttft_ms=90.0, postprocess_ms=10.0
        )
    )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            autoregressive_ttfa_ms,
            {
                "output_tokens": 2,
                "ttft_ms": math.nan,
                "decode_tokens_per_second": 100.0,
            },
        ),
        (
            required_decode_tokens_per_second,
            {
                "output_tokens": 2,
                "deadline_ms": 500.0,
                "ttft_ms": 100.0,
                "postprocess_ms": math.inf,
            },
        ),
    ],
)
def test_action_latency_equations_reject_non_finite_fixed_costs(function, kwargs):
    with pytest.raises(ValueError):
        function(**kwargs)


def test_action_latency_equations_reject_non_finite_computed_results():
    with pytest.raises(ValueError, match="computed TTFA"):
        autoregressive_ttfa_ms(
            1,
            ttft_ms=1e308,
            decode_tokens_per_second=0.0,
            postprocess_ms=1e308,
        )
    with pytest.raises(ValueError, match="finite float range"):
        required_decode_tokens_per_second(
            10**400,
            deadline_ms=1.0,
            ttft_ms=0.0,
        )


def test_action_latency_equations_require_integral_token_counts():
    with pytest.raises(ValueError):
        autoregressive_ttfa_ms(
            1.5,  # type: ignore[arg-type]
            ttft_ms=100.0,
            decode_tokens_per_second=100.0,
        )
    with pytest.raises(ValueError):
        required_decode_tokens_per_second(
            True,  # type: ignore[arg-type]
            deadline_ms=500.0,
            ttft_ms=100.0,
        )


def test_required_rate_is_inverse_of_autoregressive_ttfa():
    rate = required_decode_tokens_per_second(
        64, deadline_ms=750.0, ttft_ms=125.0, postprocess_ms=15.0
    )
    assert autoregressive_ttfa_ms(
        64,
        ttft_ms=125.0,
        decode_tokens_per_second=rate,
        postprocess_ms=15.0,
    ) == pytest.approx(750.0)


def test_rate_calibration_evaluates_200_400_600_without_fake_aggregation():
    got = calibrate_autoregressive_rate_scenarios(
        [16, 32, 64, 128],
        decode_rates_tps=[200, 400, 600],
        deadlines_ms=[500, 1000],
        predecode_ms=200,
        postprocess_ms=20,
    )
    assert got["empirical_measurement"] is False
    assert got["scenario_weighting"] is None
    assert got["aggregate_statistics_emitted"] is False
    assert got["decode_steps_semantics"].endswith("required terminal or EOS step")

    rate_200 = got["by_decode_rate_tps"]["200"]["scenarios_by_decode_steps"]
    rate_400 = got["by_decode_rate_tps"]["400"]["scenarios_by_decode_steps"]
    rate_600 = got["by_decode_rate_tps"]["600"]["scenarios_by_decode_steps"]
    assert rate_200["16"]["ttfa_ms"] == pytest.approx(295.0)
    assert rate_200["64"]["ttfa_ms"] == pytest.approx(535.0)
    assert rate_200["64"]["deadline_attainment_ms"]["500"]["meets_deadline"] is False
    assert rate_400["64"]["ttfa_ms"] == pytest.approx(377.5)
    assert rate_400["128"]["deadline_attainment_ms"]["500"]["meets_deadline"] is False
    assert rate_600["128"]["ttfa_ms"] == pytest.approx(431.6666666667)
    assert rate_600["128"]["deadline_attainment_ms"]["500"]["meets_deadline"] is True
    assert rate_200["128"]["deadline_attainment_ms"]["1000"]["meets_deadline"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decode_steps": []}, "at least one decode step"),
        ({"decode_steps": [16, 16]}, "duplicate decode step"),
        ({"decode_steps": [16], "decode_rates_tps": []}, "at least one decode rate"),
        ({"decode_steps": [16], "decode_rates_tps": [200, 200.0]}, "duplicate decode rate"),
        ({"decode_steps": [16], "decode_rates_tps": [0]}, "positive"),
    ],
)
def test_rate_calibration_rejects_ambiguous_or_empty_scenario_grids(kwargs, message):
    with pytest.raises(ValueError, match=message):
        calibrate_autoregressive_rate_scenarios(
            deadlines_ms=[500],
            predecode_ms=200,
            postprocess_ms=20,
            **kwargs,
        )


def test_summary_couples_deadline_with_exact_and_schema_valid_action():
    rows = [
        {"ttfa_ms": 80.0, "success": True, "schema_valid": True},
        {"ttfa_ms": 120.0, "success": False, "schema_valid": True},
        {"ttfa_ms": 700.0, "success": True, "schema_valid": False},
    ]
    got = summarize_action_records(rows, deadlines_ms=(100.0, 500.0, 1000.0))
    assert got["sample_count"] == 3
    assert got["total_measured_ms"] == 900.0
    assert got["exact_action_accuracy"] == pytest.approx(2 / 3)
    assert got["schema_valid_rate"] == pytest.approx(2 / 3)
    assert got["deadline_attainment_ms"]["100"]["on_time"] == 1
    assert got["deadline_attainment_ms"]["100"]["opportunities"] == 3
    assert got["deadline_attainment_ms"]["100"]["useful"] == 1
    assert got["deadline_attainment_ms"]["500"]["on_time"] == 2
    assert got["deadline_attainment_ms"]["500"]["useful"] == 1
    assert got["deadline_attainment_ms"]["1000"]["useful"] == 1
    assert got["deadline_attainment_ms"]["1000"]["success_at_deadline"] == pytest.approx(1 / 3)


def test_summary_does_not_invent_success_for_latency_only_records():
    got = summarize_action_records(
        [{"ttfa_ms": 80.0}, {"ttfa_ms": 120.0}], deadlines_ms=(100.0,)
    )
    assert "exact_action_accuracy" not in got
    assert "schema_valid_rate" not in got
    assert got["deadline_attainment_ms"]["100"] == {
        "deadline_ms": 100.0,
        "opportunities": 2,
        "on_time": 1,
        "on_time_rate": 0.5,
    }


@pytest.mark.parametrize(
    "rows",
    [
        [{"ttfa_ms": 10.0, "success": True}, {"ttfa_ms": 20.0}],
        [
            {"ttfa_ms": 10.0, "success": "false"},
            {"ttfa_ms": 20.0, "success": True},
        ],
        [
            {"ttfa_ms": 10.0, "schema_valid": True},
            {"ttfa_ms": 20.0, "schema_valid": 1},
        ],
    ],
)
def test_summary_rejects_partial_or_non_boolean_quality_columns(rows):
    with pytest.raises(ValueError):
        summarize_action_records(rows)


def test_summary_rejects_invalid_latency_and_deadlines():
    with pytest.raises(ValueError, match="record 1"):
        summarize_action_records([{"ttfa_ms": 1.0}, {}])
    with pytest.raises(ValueError, match="finite"):
        summarize_action_records([{"ttfa_ms": math.nan}])
    with pytest.raises(ValueError, match="JSON number"):
        summarize_action_records([{"ttfa_ms": "1.0"}])  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="mapping"):
        summarize_action_records([["ttfa_ms", 1.0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        summarize_action_records([{"ttfa_ms": 1.0}], deadlines_ms=(math.nan,))
    with pytest.raises(ValueError, match="duplicate deadline"):
        summarize_action_records([{"ttfa_ms": 1.0}], deadlines_ms=(100, 100.0))
    with pytest.raises(ValueError, match="at least one deadline"):
        summarize_action_records([{"ttfa_ms": 1.0}], deadlines_ms=())


def test_distinct_close_deadlines_keep_distinct_json_keys():
    got = summarize_action_records(
        [{"ttfa_ms": 100.000015}],
        deadlines_ms=(100.00001, 100.00002),
    )
    assert len(got["deadline_attainment_ms"]) == 2


def test_zero_duration_goodput_is_json_safe():
    got = summarize_action_records(
        [{"ttfa_ms": 0.0, "success": True, "schema_valid": True}],
        deadlines_ms=(100.0,),
    )
    assert got["deadline_attainment_ms"]["100"]["useful_actions_per_minute"] is None


def test_grouped_summary_surfaces_per_family_quality():
    rows = [
        {
            "family": "computer_use",
            "ttfa_ms": 40.0,
            "success": True,
            "schema_valid": True,
        },
        {
            "family": "text",
            "ttfa_ms": 20.0,
            "success": False,
            "schema_valid": True,
        },
    ]
    got = summarize_grouped_action_records(rows, deadlines_ms=(100.0,))
    assert list(got) == ["computer_use", "text"]
    assert got["computer_use"]["exact_action_accuracy"] == 1.0
    assert got["text"]["exact_action_accuracy"] == 0.0
    with pytest.raises(ValueError, match="record 1"):
        summarize_grouped_action_records([rows[0], {"ttfa_ms": 20.0}])


def test_case_cluster_bootstrap_is_deterministic_and_repeat_invariant():
    rows = [
        *_case_rows("case-a", [True]),
        *_case_rows("case-b", [True, False]),
        *_case_rows("case-c", [False]),
        *_case_rows("case-d", [True]),
    ]
    got = bootstrap_success_at_deadline_ci(
        rows,
        deadline_ms=100.0,
        resamples=2_000,
        seed=17,
    )
    assert got == bootstrap_success_at_deadline_ci(
        list(reversed(rows)),
        deadline_ms=100.0,
        resamples=2_000,
        seed=17,
    )
    assert got["estimate"] == pytest.approx(0.625)
    assert got["opportunity_estimate"] == pytest.approx(0.6)
    assert got["lower"] == 0.25
    assert got["upper"] == 1.0
    assert got["cluster_count"] == 4
    assert got["resamples"] == 2_000
    assert got["seed"] == 17
    assert got["estimand"] == "mean_case_success_at_deadline"

    duplicated = [row for row in rows for _ in range(20)]
    assert bootstrap_success_at_deadline_ci(
        duplicated,
        deadline_ms=100.0,
        resamples=2_000,
        seed=17,
    ) == got

    with pytest.raises(ValueError, match="at least two clusters"):
        bootstrap_success_at_deadline_ci(
            _case_rows("only-case", [True] * 10),
            deadline_ms=100.0,
            resamples=100,
        )


def test_summary_adds_case_bootstrap_ci_without_treating_repeats_as_tasks():
    rows = [
        *_case_rows("case-a", [True] * 50),
        *_case_rows("case-b", [False] * 50),
    ]
    got = summarize_action_records(
        rows,
        deadlines_ms=(100.0,),
        bootstrap_resamples=250,
        bootstrap_seed=9,
    )
    ci = got["deadline_attainment_ms"]["100"]["case_macro_success_at_deadline_ci95"]
    assert ci["estimate"] == 0.5
    assert ci["cluster_count"] == 2
    assert ci["resamples"] == 250
    assert (
        ci["estimate"]
        == got["deadline_attainment_ms"]["100"]["case_macro_success_at_deadline"]
    )
    assert 0.0 <= ci["lower"] <= ci["estimate"] <= ci["upper"] <= 1.0

    with pytest.raises(ValueError, match="cluster field"):
        summarize_action_records(
            [{"ttfa_ms": 10.0, "success": True, "schema_valid": True}],
            deadlines_ms=(100.0,),
            bootstrap_resamples=10,
        )


def test_summary_labels_unbalanced_opportunity_and_case_macro_scores_separately():
    rows = [
        *_case_rows("case-a", [True]),
        *_case_rows("case-b", [True, False]),
        *_case_rows("case-c", [False]),
        *_case_rows("case-d", [True]),
    ]
    got = summarize_action_records(
        rows,
        deadlines_ms=(100.0,),
        bootstrap_resamples=250,
        bootstrap_seed=9,
    )
    deadline = got["deadline_attainment_ms"]["100"]
    assert deadline["success_at_deadline"] == pytest.approx(0.6)
    assert deadline["case_macro_success_at_deadline"] == pytest.approx(0.625)
    assert (
        deadline["case_macro_success_at_deadline_ci95"]["opportunity_estimate"]
        == pytest.approx(0.6)
    )
    assert "success_at_deadline_ci95" not in deadline


def test_paired_case_bootstrap_reports_candidate_minus_baseline_delta():
    baseline = [
        *_case_rows("case-a", [False, False]),
        *_case_rows("case-b", [False]),
        *_case_rows("case-c", [True]),
    ]
    candidate = [
        *_case_rows("case-a", [True, True]),
        *_case_rows("case-b", [False]),
        *_case_rows("case-c", [True]),
    ]
    got = paired_success_at_deadline_delta_ci(
        baseline,
        candidate,
        deadline_ms=100.0,
        resamples=1_000,
        seed=4,
    )
    assert got["estimate"] == pytest.approx(1 / 3)
    assert got["baseline_estimate"] == pytest.approx(1 / 3)
    assert got["candidate_estimate"] == pytest.approx(2 / 3)
    assert got["direction"] == "candidate_minus_baseline"
    assert got["estimand"] == "mean_paired_case_success_delta"
    assert got["lower"] == 0.0
    assert got["upper"] == 1.0

    with pytest.raises(ValueError, match="identical case IDs"):
        paired_success_at_deadline_delta_ci(
            baseline,
            candidate[:-1],
            deadline_ms=100.0,
            resamples=10,
        )


def test_paired_bootstrap_distinguishes_case_macro_and_opportunity_deltas():
    baseline = [
        *_case_rows("case-a", [False]),
        *_case_rows("case-b", [True, True, True]),
    ]
    candidate = [
        *_case_rows("case-a", [True]),
        *_case_rows("case-b", [True, True, False]),
    ]
    got = paired_success_at_deadline_delta_ci(
        baseline,
        candidate,
        deadline_ms=100.0,
        resamples=500,
        seed=7,
    )
    assert got["baseline_estimate"] == 0.5
    assert got["candidate_estimate"] == pytest.approx(5 / 6)
    assert got["estimate"] == pytest.approx(1 / 3)
    assert got["baseline_opportunity_estimate"] == 0.75
    assert got["candidate_opportunity_estimate"] == 0.75
    assert got["opportunity_delta"] == 0.0
    assert got["opportunity_count_per_system"] == 4

    with pytest.raises(ValueError, match="same number of opportunities per case"):
        paired_success_at_deadline_delta_ci(
            [*_case_rows("case-a", [False, False]), *_case_rows("case-b", [True])],
            [*_case_rows("case-a", [True]), *_case_rows("case-b", [True])],
            deadline_ms=100.0,
            resamples=10,
        )


def test_cli_summarizes_browser_export_shape(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "benchmark": "localagent-held-out-action-latency",
        "metadata": {
            "backend": "webgpu",
            "concurrency": 1,
            "cases": 4,
            "repetitions": 1,
            "latency_clock": "harness_ttfa_ms",
        },
        # The Python CLI deliberately recomputes the summary from raw records.
        "summary": {"stale_browser_summary": True},
        "records": [
            {
                "case_id": "gui-click",
                "family": "computer_use",
                "repetition": 0,
                "backend": "webgpu",
                "tokenize_ms": 1.0,
                "inference_ms": 37.0,
                "decode_control_ms": 0.0,
                "dispatch_ms": 1.0,
                "parse_validate_ms": 1.0,
                "ttft_ms": 10.0,
                "tpot_ms": 2.0,
                "harness_ttfa_ms": 40.0,
                "ttfa_ms": 40.0,
                "schema_valid": True,
                "success": True,
            },
            {
                "case_id": "gui-click-2",
                "family": "computer_use",
                "repetition": 0,
                "backend": "webgpu",
                "tokenize_ms": 1.0,
                "inference_ms": 37.0,
                "decode_control_ms": 0.0,
                "dispatch_ms": 1.0,
                "parse_validate_ms": 1.0,
                "ttft_ms": 10.0,
                "tpot_ms": 2.0,
                "harness_ttfa_ms": 40.0,
                "ttfa_ms": 40.0,
                "schema_valid": True,
                "success": True,
            },
            {
                "case_id": "abstain",
                "family": "text",
                "repetition": 0,
                "backend": "webgpu",
                "tokenize_ms": 1.0,
                "inference_ms": 597.0,
                "decode_control_ms": 0.0,
                "dispatch_ms": 1.0,
                "parse_validate_ms": 1.0,
                "ttft_ms": 20.0,
                "tpot_ms": None,
                "harness_ttfa_ms": 600.0,
                "ttfa_ms": 600.0,
                "schema_valid": True,
                "success": False,
            },
            {
                "case_id": "abstain-2",
                "family": "text",
                "repetition": 0,
                "backend": "webgpu",
                "tokenize_ms": 1.0,
                "inference_ms": 597.0,
                "decode_control_ms": 0.0,
                "dispatch_ms": 1.0,
                "parse_validate_ms": 1.0,
                "ttft_ms": 20.0,
                "tpot_ms": None,
                "harness_ttfa_ms": 600.0,
                "ttfa_ms": 600.0,
                "schema_valid": True,
                "success": False,
            },
        ],
    }
    path = tmp_path / "browser-export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "realtime_agent_benchmark.py"),
            "summarize",
            str(path),
            "--deadlines",
            "500,1000",
            "--bootstrap-resamples",
            "200",
            "--bootstrap-seed",
            "23",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    got = json.loads(completed.stdout)
    assert got["sample_count"] == 4
    assert got["latency_key"] == "harness_ttfa_ms"
    assert got["deadline_attainment_ms"]["500"]["success_at_deadline"] == 0.5
    assert got["deadline_attainment_ms"]["1000"]["on_time_rate"] == 1.0
    assert got["stage_latency_ms"]["inference_ms"]["p50"] == 317.0
    assert got["stage_latency_ms"]["ttft_ms"]["p50"] == 15.0
    assert got["stage_latency_ms"]["tpot_ms"]["count"] == 2
    assert got["stage_latency_ms"]["tpot_ms"]["missing_count"] == 2
    assert got["by_family"]["computer_use"]["exact_action_accuracy"] == 1.0
    assert got["by_family"]["text"]["exact_action_accuracy"] == 0.0
    assert got["export_integrity"]["declared_opportunity_count"] == 4
    assert got["export_integrity"]["record_count_matches"] is True
    assert got["export_integrity"]["selected_clock_matches_declaration"] is True
    ci = got["deadline_attainment_ms"]["500"]["case_macro_success_at_deadline_ci95"]
    assert ci["cluster_count"] == 4
    assert ci["resamples"] == 200
    assert ci["seed"] == 23


def test_cli_emits_labeled_200_400_600_rate_counterfactual():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "realtime_agent_benchmark.py"),
            "calibrate",
            "--decode-steps",
            "64,128",
            "--decode-rates",
            "200,400,600",
            "--deadlines",
            "500,1000",
            "--ttft-ms",
            "200",
            "--parse-ms",
            "20",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    got = json.loads(completed.stdout)
    assert got["artifact_type"] == "autoregressive_decode_rate_counterfactual"
    assert got["empirical_measurement"] is False
    assert got["decode_rates_tps"] == [200.0, 400.0, 600.0]
    assert (
        got["by_decode_rate_tps"]["200"]["scenarios_by_decode_steps"]["64"][
            "deadline_attainment_ms"
        ]["500"]["meets_deadline"]
        is False
    )
    assert (
        got["by_decode_rate_tps"]["600"]["scenarios_by_decode_steps"]["128"][
            "deadline_attainment_ms"
        ]["500"]["meets_deadline"]
        is True
    )


def test_cli_retains_runtime_failure_rows_with_nullable_stage_timings(tmp_path: Path):
    payload = {
        "metadata": {
            "concurrency": 1,
            "cases": 1,
            "repetitions": 1,
            "latency_clock": "harness_ttfa_ms",
        },
        "records": [
            {
                "case_id": "failed-case",
                "family": "computer_use",
                "repetition": 0,
                "harness_ttfa_ms": 375.0,
                "runtime_ttfa_ms": 375.0,
                "tokenize_ms": None,
                "inference_ms": None,
                "decode_control_ms": None,
                "dispatch_ms": None,
                "parse_validate_ms": None,
                "ttft_ms": None,
                "tpot_ms": None,
                "success": False,
                "schema_valid": False,
            }
        ],
    }
    path = tmp_path / "failed-export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "realtime_agent_benchmark.py"),
            "summarize",
            str(path),
            "--deadlines",
            "500",
            "--bootstrap-resamples",
            "0",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    got = json.loads(completed.stdout)
    deadline = got["deadline_attainment_ms"]["500"]
    assert deadline["opportunities"] == 1
    assert deadline["on_time"] == 1
    assert deadline["success_at_deadline"] == 0.0
    assert got["stage_latency_ms"]["inference_ms"] == {
        "count": 0,
        "missing_count": 1,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "metadata": {
                    "concurrency": 1,
                    "cases": 2,
                    "repetitions": 1,
                },
                "records": [
                    {
                        "case_id": "case-a",
                        "family": "tool",
                        "repetition": 0,
                        "ttfa_ms": 10.0,
                        "success": False,
                        "schema_valid": False,
                    }
                ],
            },
            "failed and invalid trials must remain",
        ),
        (
            [
                {
                    "case_id": "case-a",
                    "ttfa_ms": 10.0,
                    "success": False,
                    "schema_valid": False,
                },
                {
                    "case_id": "case-b",
                    "ttfa_ms": 10.0,
                    "success": False,
                    "schema_valid": False,
                },
            ],
            "group field 'family' must be present",
        ),
    ],
)
def test_cli_rejects_incomplete_paper_reporting(
    tmp_path: Path,
    payload: object,
    message: str,
):
    path = tmp_path / "invalid-export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "realtime_agent_benchmark.py"),
            "summarize",
            str(path),
            "--bootstrap-resamples",
            "10",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode != 0
    assert message in completed.stderr
