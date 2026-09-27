from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from localagent.eval.webgpu_dom_summary import (
    _validate_record,
    build_webgpu_dom_summary,
    write_webgpu_dom_summary,
)
from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RAW_PATHS = [
    ROOT
    / "docs/paper/results/raw"
    / f"m5-webgpu-sft-dom-pilot-seed2027-run{run}.json"
    for run in (1, 2, 3)
]
SUMMARY_PATH = ROOT / "docs/paper/results/m5-webgpu-sft-dom-pilot-seed2027.summary.json"
EXPECTED_RAW_IDENTITIES = (
    (
        1_234_269,
        "77f2e4aa170a54d2f8435ee00743978df4095d045e28a9f7fb8b4e41f1c253f9",
    ),
    (
        1_232_271,
        "0f7ba7a63b02182416fced37a1d5321c8700b34ff4f97c0661a3f60da3aca9e5",
    ),
    (
        1_232_090,
        "88b0dd770e938b980d04cc25fc0b4d2bbf0732978698cfedf6035d3058133e2f",
    ),
)


def test_tracked_webgpu_dom_runs_reproduce_aggregate(tmp_path: Path) -> None:
    summary = build_webgpu_dom_summary(RAW_PATHS, repository_root=ROOT)

    assert summary == json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["validation"]["status"] == "mechanically_valid"
    assert all(summary["validation"]["checks"].values())
    assert [
        (run["raw_artifact"]["bytes"], run["raw_artifact"]["sha256"])
        for run in summary["runs"]
    ] == list(EXPECTED_RAW_IDENTITIES)

    protocol = summary["protocol"]
    assert protocol["raw_schema_version"] == 2
    assert protocol["benchmark"] == "localagent-single-step-dom-microtasks"
    assert protocol["benchmark_version"] == "rtab-dom-0.2"
    assert protocol["backend"] == "webgpu"
    assert protocol["execution_provider_request"]["requested"] == "webgpu"
    assert protocol["execution_provider_request"]["session_provider_count"] == 1
    assert protocol["execution_provider_request"]["whole_session_retry"] is False
    assert protocol["onnxruntime_web_version"] == "1.27.0"
    assert protocol["policy"] == "structured_one_forward"
    assert protocol["inference_passes_per_record"] == 1
    assert protocol["target_input_tokens"] == 512
    assert protocol["warmups_per_run"] == 3
    assert protocol["cases"] == 8
    assert protocol["repetitions_per_case"] == 30
    assert protocol["records_per_run"] == 240
    assert protocol["case_order_seed"] == "dom-loop-v1"
    assert protocol["success_at_deadline_definition"] == (
        "score.exact_action && score.schema_valid && harness_ttfa_ms <= deadline"
    )
    assert protocol["success_at_deadline_includes_dom_success"] is False
    assert protocol["dom_success_metric"] == "score.closed_loop_success"
    assert protocol["record_outcome_evidence"] == {
        "mode": "independently_recomputed",
        "exact_action": (
            "recomputed from historical raw_model_output and full expected action; stored "
            "exact-tool, exact-args, exact-action, and success booleans must agree"
        ),
        "schema_validity": (
            "recomputed from historical raw_model_output and "
            "independent_schema.tool_schema; stored validator result, errors, "
            "schema-validity, and score aliases must agree"
        ),
        "historical_interpretation": (
            "rtab-dom-0.2 raw_model_output is used only as its original structured action "
            "object; no generated-text or post-hoc model-output reinterpretation is performed"
        ),
    }

    identity = summary["identity"]
    assert identity["checkpoint"] == {
        "sha256": "79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1",
        "stage": "sft",
        "step": 319,
    }
    assert identity["graph"] == {
        "file": "action_model.fp16.onnx",
        "bytes": 21_430_301,
        "sha256": "b91e7f84077155640a5e288a7c58c2245c298859ddd86bd7268e71039e65c49a",
    }
    assert identity["tokenizer"]["sha256"] == (
        "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c"
    )
    assert identity["held_out_suite"] == {
        "file": "browser-task-cases.json",
        "bytes": 6_285,
        "sha256": "4c46b5b347257b81e716ec0a20a6c6116df716466e1ba8e8a74a117bb5708971",
        "schema_version": 1,
    }

    aggregate = summary["aggregate"]
    assert aggregate["run_count"] == 3
    assert aggregate["records"] == aggregate["abstentions"] == 720
    assert aggregate["all_predictions_abstained"] is True
    assert aggregate["score_counts"] == {
        "exact_tool": 0,
        "exact_args": 0,
        "exact_action": 0,
        "schema_valid": 0,
        "final_dom_valid": 0,
        "state_transition": 0,
        "closed_loop_success": 0,
    }
    assert set(aggregate["pooled_rates"].values()) == {0.0}

    run_level = aggregate["run_level_latency_ms"]
    assert run_level["harness_ttfa_ms"]["p50"]["by_run"] == pytest.approx(
        [27.549999982118607, 27.400000005960464, 28.450000017881393]
    )
    assert run_level["harness_ttfa_ms"]["p95"]["by_run"] == pytest.approx(
        [35.32000001072883, 35.1999999910593, 35.30500001013279]
    )
    assert run_level["closed_loop_ms"]["p50"]["by_run"] == pytest.approx(
        [33.40000003576279, 33.349999994039536, 33.30000001192093]
    )
    assert run_level["closed_loop_ms"]["p95"]["by_run"] == pytest.approx(
        [67.72000004351138, 66.70999999046325, 66.6999999910593]
    )

    pooled = aggregate["pooled_metrics"]
    assert pooled["latency_ms"]["harness_ttfa_ms"] == pytest.approx(
        {
            "count": 720,
            "min": 18.599999964237213,
            "mean": 28.30361111130979,
            "p50": 27.80000001192093,
            "p90": 34.30000001192093,
            "p95": 35.30000001192093,
            "p99": 39.72400001645086,
            "max": 114.69999998807907,
        }
    )
    assert pooled["latency_ms"]["closed_loop_ms"] == pytest.approx(
        {
            "count": 720,
            "min": 29.19999998807907,
            "mean": 39.63944444482525,
            "p50": 33.30000001192093,
            "p90": 66.39999997615814,
            "p95": 66.80000001192093,
            "p99": 68.69999998807907,
            "max": 184.5,
        }
    )
    assert pooled["deadline_attainment_ms"]["100"] == {
        "opportunities": 720,
        "on_time": 719,
        "on_time_rate": 719 / 720,
        "useful": 0,
        "success_at_deadline": 0.0,
        "useful_actions_per_minute": 0.0,
    }
    for deadline in ("250", "500", "1000", "2000"):
        assert pooled["deadline_attainment_ms"][deadline]["on_time"] == 720
        assert pooled["deadline_attainment_ms"][deadline]["success_at_deadline"] == 0.0

    unsigned = copy.deepcopy(summary)
    recorded_hash = unsigned.pop("summary_sha256")
    assert recorded_hash == canonical_sha256(unsigned)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_webgpu_dom_summary(summary, first)
    write_webgpu_dom_summary(summary, second)
    assert first.read_bytes() == second.read_bytes()


def _mutated_raw_paths(tmp_path: Path, mutation: str) -> list[Path]:
    paths: list[Path] = []
    for run_number, source in enumerate(RAW_PATHS, start=1):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if run_number == 2:
            if mutation == "records":
                payload["records"].pop()
            elif mutation == "ttfa":
                payload["records"][0]["latency_ms"]["harness_ttfa_ms"] = "not-finite"
            elif mutation == "graph":
                payload["metadata"]["graph_hash"] = "0" * 64
            elif mutation == "policy":
                payload["records"][0]["raw_model_output"]["policy"] = "autoregressive"
            elif mutation == "protocol":
                payload["metadata"]["target_input_tokens"] = 511
            else:  # pragma: no cover - test helper guard
                raise AssertionError(mutation)
        destination = tmp_path / source.name
        destination.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(destination)
    return paths


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("records", "record count must be exactly 240"),
        ("ttfa", "harness_ttfa_ms must be a finite number"),
        ("graph", "graph does not match the pilot identity"),
        ("policy", "is not one structured inference pass"),
        ("protocol", "target_input_tokens does not match the DOM pilot protocol"),
    ],
)
def test_webgpu_dom_summary_rejects_invalid_runs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _mutated_raw_paths(tmp_path, mutation)

    with pytest.raises(ValueError, match=message):
        build_webgpu_dom_summary(paths, repository_root=tmp_path)


def test_webgpu_dom_summary_writer_rejects_invalid_self_hash(tmp_path: Path) -> None:
    summary = build_webgpu_dom_summary(RAW_PATHS, repository_root=ROOT)
    summary["aggregate"]["abstentions"] = 719

    with pytest.raises(ValueError, match="self-hash"):
        write_webgpu_dom_summary(summary, tmp_path / "summary.json")


def _v04_dom_record() -> dict[str, object]:
    schema = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
        "additionalProperties": False,
    }
    predicted = {"tool": "click", "args": {"target": "Confirm"}}
    expected = {"tool": "click", "args": {"target": "Confirm"}}
    before = {
        "passed": False,
        "assertions": [
            {
                "document": "top",
                "selector": "#fixture-confirm",
                "kind": "dataset",
                "actual": "idle",
                "passed": False,
            }
        ],
    }
    after = {
        "passed": True,
        "assertions": [
            {
                "document": "top",
                "selector": "#fixture-confirm",
                "kind": "dataset",
                "actual": "confirmed",
                "passed": True,
            }
        ],
    }
    score = {
        "exact_tool": True,
        "exact_args": True,
        "exact_action": True,
        "schema_valid": True,
        "app_schema_valid_diagnostic": True,
        "schema_validator_agreement": True,
        "fixture_clean": True,
        "execution_ok": True,
        "final_dom_valid": True,
        "state_transition": True,
        "closed_loop_success": True,
    }
    latency = {
        "harness_ttfa_ms": 25.0,
        "runtime_ttfa_ms": 24.5,
        "independent_validate_ms": 0.5,
        "model_wall_ms": 24.5,
        "tool_ms": 1.0,
        "paint_wait_ms": 2.0,
        "closed_loop_ms": 28.0,
    }
    return {
        "case_id": "confirm-click",
        "family": "click",
        "fixture": {"id": "confirm", "version": 1},
        "query": "Click Confirm.",
        "expected": expected,
        "expected_action": expected,
        "predicted_action": predicted,
        "repetition": 0,
        "order_index": 0,
        "measured": True,
        "backend": "webgpu",
        "input_tokens": 512,
        "natural_input_tokens": 16,
        "context_padding_tokens": 496,
        "context_padding_placement": "after_natural_assistant_marker",
        "decision_input_tokens": 16,
        "decision_feature_index": 15,
        "action_timeout_ms": 10_000,
        "watchdog_outcome": "completed_before_timeout",
        "parse_evidence": {
            "policy": "structured_one_forward",
            "inference_passes": 1,
            "parse_kind": "structured_heads",
            "parse_failure": False,
            "parse_error": None,
            "runtime_validation_failure": False,
            "runtime_validation_error": None,
            "runtime_error": None,
        },
        "parse_failure": False,
        "validation_failure": False,
        "independent_schema": {
            "validator": "browser-task-json-schema-subset-v2",
            "valid": True,
            "errors": [],
            "schema_tool": "click",
            "tool_schema": schema,
        },
        "execution": {"ok": True},
        "dom_before": before,
        "dom_after": after,
        "score": score,
        "success": True,
        "schema_valid": True,
        "predicted_tool": "click",
        "expected_tool": "click",
        "latency_ms": latency,
        "harness_ttfa_ms": 25.0,
        "runtime_ttfa_ms": 24.5,
        "independent_validate_ms": 0.5,
        "ttfa_ms": 25.0,
    }


def test_webgpu_dom_v04_record_scores_recompute_from_raw_evidence() -> None:
    _validate_record(
        _v04_dom_record(),
        label="record",
        measured=True,
        expected_repetition=0,
        expected_order=0,
        benchmark_version="rtab-dom-0.4",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("score", "score.exact_action differs"),
        ("expected_args", "score.exact_args differs"),
        ("schema_boolean", "independent_schema does not reproduce"),
        ("schema_errors", "independent_schema does not reproduce"),
        ("root_schema", "root score aliases differ"),
        ("schema_diagnostic", "schema-validator diagnostics differ"),
        ("watchdog", "watchdog evidence differs"),
    ],
)
def test_webgpu_dom_v04_record_rejects_tampered_evidence(
    mutation: str,
    message: str,
) -> None:
    record = _v04_dom_record()
    if mutation == "score":
        record["score"]["exact_action"] = False  # type: ignore[index]
    elif mutation == "expected_args":
        record["expected_action"]["args"]["target"] = "Cancel"  # type: ignore[index]
        record["expected"]["args"]["target"] = "Cancel"  # type: ignore[index]
    elif mutation == "schema_boolean":
        record["independent_schema"]["valid"] = False  # type: ignore[index]
    elif mutation == "schema_errors":
        record["independent_schema"]["errors"] = ["tampered"]  # type: ignore[index]
    elif mutation == "root_schema":
        record["schema_valid"] = False
    elif mutation == "schema_diagnostic":
        record["score"]["schema_validator_agreement"] = False  # type: ignore[index]
    elif mutation == "watchdog":
        record["action_timeout_ms"] = 0
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        _validate_record(
            record,
            label="record",
            measured=True,
            expected_repetition=0,
            expected_order=0,
            benchmark_version="rtab-dom-0.4",
        )
