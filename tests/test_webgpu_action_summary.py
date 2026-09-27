from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from localagent.eval.webgpu_action_summary import (
    _validate_v04_action_evidence,
    build_webgpu_action_summary,
    write_webgpu_action_summary,
)
from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RAW_PATHS = [
    ROOT
    / "docs/paper/results/raw"
    / f"m5-webgpu-sft-action-pilot-seed2027-run{run}.json"
    for run in (1, 2, 3)
]
SUMMARY_PATH = (
    ROOT / "docs/paper/results/m5-webgpu-sft-action-pilot-seed2027.summary.json"
)
EXPECTED_RAW_IDENTITIES = (
    (
        776_941,
        "95d2cf9648b4b34338a3b03a57d6f716103c5c55a43f68c6151682a55e562356",
    ),
    (
        776_822,
        "053f0997d8bcf7e04aad809010711170f6d3aeaaf0da11844c648734da41ca27",
    ),
    (
        776_494,
        "f3ddb777090e1278d1e56cbe112fcda88a6b287d27bd5135a7617e7fe76feff9",
    ),
)


def test_tracked_webgpu_action_runs_reproduce_aggregate(tmp_path: Path) -> None:
    summary = build_webgpu_action_summary(RAW_PATHS, repository_root=ROOT)

    assert summary == json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["validation"]["status"] == "mechanically_valid"
    assert all(summary["validation"]["checks"].values())
    assert [
        (run["raw_artifact"]["bytes"], run["raw_artifact"]["sha256"])
        for run in summary["runs"]
    ] == list(EXPECTED_RAW_IDENTITIES)
    assert summary["protocol"] == {
        "raw_schema_version": 3,
        "benchmark": "localagent-held-out-action-latency",
        "benchmark_version": "rtab-0.2",
        "backend": "webgpu",
        "backend_requirement": "explicit-webgpu-no-whole-session-retry",
        "execution_provider_request": {
            "requested": "webgpu",
            "session_provider_count": 1,
            "whole_session_retry": False,
            "single_provider_session_creation_succeeded": True,
            "per_node_placement": "unknown",
            "per_node_fallback_status": "unknown",
            "note": (
                "ORT Web does not expose per-node placement; this proves the requested session "
                "provider and does not claim every node executed on the GPU."
            ),
        },
        "onnxruntime_web_version": "1.27.0",
        "browser_user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        ),
        "gpu_adapter": {
            "vendor": "apple",
            "architecture": "metal-3",
            "device": None,
            "description": None,
            "is_fallback_adapter": False,
        },
        "policy": "structured_one_forward",
        "decode_strategy": "one_forward_structured_heads",
        "precision": "fp16",
        "target_input_tokens": 512,
        "warmups_per_run": 3,
        "cases": 20,
        "repetitions_per_case": 30,
        "opportunities_per_run": 600,
        "case_order_seed": "slmw2026-v1",
        "concurrency": 1,
        "latency_clock": "harness_ttfa_ms",
        "latency_boundaries": {
            "harness_ttfa": (
                "immediately before prompt tokenization through independent schema validation"
            ),
            "runtime_ttfa": (
                "runtime prompt tokenization through runtime parse/schema validation"
            ),
            "ttfa_ms": (
                "backward-compatible exact alias of harness_ttfa_ms; not an additional clock"
            ),
            "ttft": "runtime inference submission through first sampled token",
            "exact_action_scoring": "excluded from TTFA",
        },
        "deadlines_ms": [100, 250, 500, 1000, 2000],
        "record_outcome_evidence": {
            "mode": "browser_reported_non_recomputable",
            "exact_action": (
                "historical rtab-0.2 rows omit predicted args and full expected actions, so "
                "exact action outcomes are browser-reported and cannot be independently "
                "recomputed"
            ),
            "schema_validity": (
                "historical rtab-0.2 rows omit validator errors and selected tool schemas, so "
                "schema validity is browser-reported and cannot be independently recomputed"
            ),
        },
    }
    assert summary["identity"]["checkpoint"] == {
        "sha256": "79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1",
        "stage": "sft",
        "step": 319,
    }
    assert summary["identity"]["graph"] == {
        "file": "action_model.fp16.onnx",
        "bytes": 21_430_301,
        "sha256": "b91e7f84077155640a5e288a7c58c2245c298859ddd86bd7268e71039e65c49a",
    }
    assert summary["identity"]["bundle_manifest"] == {
        "raw_bytes": 9_294,
        "raw_sha256": (
            "86bbee00d783ca69af02843a4cf935ff978612b81b6a2fedd47fd943e611bee4"
        ),
        "canonical_sha256": (
            "5fee08dfaf4dab4a4d58f506c3fe55ba38c7168ea929f316f307351db7be3fd5"
        ),
        "schema_version": 3,
    }

    aggregate = summary["aggregate"]
    assert aggregate["run_count"] == 3
    assert aggregate["opportunities"] == 1_800
    assert aggregate["exact_actions"] == 90
    assert aggregate["schema_valid_actions"] == 1_800
    assert aggregate["parse_failures"] == aggregate["validation_failures"] == 0
    assert aggregate["outcome_breakdown"] == {
        "expected_abstention_opportunities": 90,
        "expected_tool_opportunities": 1_710,
        "null_predicted_tool": 1_800,
        "non_null_predicted_tool": 0,
        "exact_expected_abstentions": 90,
        "exact_expected_tools": 0,
        "exact_rate_expected_abstentions": 1.0,
        "exact_rate_expected_tools": 0.0,
    }
    assert aggregate["run_level_ttfa_ms"]["p50"]["by_run"] == pytest.approx(
        [24.750000029802322, 24.549999982118607, 25.19999998807907]
    )
    assert aggregate["run_level_ttfa_ms"]["p95"]["by_run"] == pytest.approx(
        [34.40499997735023, 34.30000001192093, 34.80000001192093]
    )
    pooled = aggregate["pooled_metrics"]
    assert pooled["latency_ms"]["harness_ttfa_ms"] == pytest.approx(
        {
            "count": 1_800,
            "min": 15.0,
            "mean": 25.667666666805744,
            "p50": 24.80000001192093,
            "p90": 32.0,
            "p95": 34.39999997615814,
            "p99": 38.90099997639656,
            "max": 53.5,
        }
    )
    assert pooled["exact_action_accuracy"] == 0.05
    assert pooled["schema_valid_rate"] == 1.0
    for deadline in ("100", "250", "500", "1000", "2000"):
        result = pooled["deadline_attainment_ms"][deadline]
        assert result["opportunities"] == result["on_time"] == 1_800
        assert result["useful"] == 90
        assert result["success_at_deadline"] == 0.05

    unsigned = copy.deepcopy(summary)
    recorded_sha256 = unsigned.pop("summary_sha256")
    assert recorded_sha256 == canonical_sha256(unsigned)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_webgpu_action_summary(summary, first)
    write_webgpu_action_summary(summary, second)
    assert first.read_bytes() == second.read_bytes()


def _mutated_raw_paths(tmp_path: Path, mutation: str) -> list[Path]:
    paths: list[Path] = []
    for run_number, source in enumerate(RAW_PATHS, start=1):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if run_number == 2:
            if mutation == "opportunities":
                payload["records"].pop()
            elif mutation == "ttfa":
                payload["records"][0]["harness_ttfa_ms"] = "not-finite"
            elif mutation == "checkpoint":
                payload["metadata"]["checkpoint_hash"] = "0" * 64
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
        ("opportunities", "opportunity count must be exactly 600"),
        ("ttfa", "harness_ttfa_ms must be a finite number"),
        ("checkpoint", "checkpoint and manifest identities differ"),
        ("protocol", "target_input_tokens does not match the pilot protocol"),
    ],
)
def test_webgpu_action_summary_rejects_invalid_runs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _mutated_raw_paths(tmp_path, mutation)

    with pytest.raises(ValueError, match=message):
        build_webgpu_action_summary(paths, repository_root=tmp_path)


def test_webgpu_action_summary_writer_rejects_invalid_self_hash(tmp_path: Path) -> None:
    summary = build_webgpu_action_summary(RAW_PATHS, repository_root=ROOT)
    summary["aggregate"]["exact_actions"] = 91

    with pytest.raises(ValueError, match="self-hash"):
        write_webgpu_action_summary(summary, tmp_path / "summary.json")


def _v04_action_record() -> dict[str, object]:
    schema = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
        "additionalProperties": False,
    }
    return {
        "predicted_action": {"tool": "click", "args": {"target": "Confirm"}},
        "expected_action": {"tool": "click", "args": {"target": "Confirm"}},
        "independent_schema": {
            "validator": "benchmark-json-schema-subset-v2",
            "valid": True,
            "errors": [],
            "schema_tool": "click",
            "tool_schema": schema,
        },
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
        "exact_tool": True,
        "exact_args": True,
        "exact_action": True,
        "success": True,
        "schema_valid": True,
        "validation_failure": False,
        "parse_failure": False,
        "predicted_tool": "click",
        "expected_tool": "click",
        "action_timeout_ms": 10_000,
        "watchdog_outcome": "completed_before_timeout",
    }


def test_webgpu_action_v04_record_scores_recompute_from_raw_evidence() -> None:
    _validate_v04_action_evidence(_v04_action_record(), "record")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("success", "success disagrees with raw actions"),
        ("args", "exact_args disagrees with raw actions"),
        ("schema_boolean", "schema_valid disagrees with independent schema evidence"),
        ("schema_errors", "does not reproduce from raw action and schema"),
        ("watchdog", "watchdog evidence differs"),
    ],
)
def test_webgpu_action_v04_record_rejects_tampered_evidence(
    mutation: str,
    message: str,
) -> None:
    record = _v04_action_record()
    if mutation == "success":
        record["success"] = False
    elif mutation == "args":
        record["expected_action"] = {"tool": "click", "args": {"target": "Cancel"}}
    elif mutation == "schema_boolean":
        record["schema_valid"] = False
    elif mutation == "schema_errors":
        record["independent_schema"]["errors"] = ["tampered"]  # type: ignore[index]
    elif mutation == "watchdog":
        record["watchdog_outcome"] = "timed_out_but_continued"
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        _validate_v04_action_evidence(record, "record")
