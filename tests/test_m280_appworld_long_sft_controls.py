"""Integrity checks for the longer AppWorld code-SFT/free-run controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/paper/results/raw"
SFT = RAW / "m280-appworld-long-sft-v1.json"
RUNPYTHON = RAW / "m280-appworld-long-native-runpython-v1.json"
COMBINED = RAW / "m280-appworld-long-heads-native-v1.json"
EXACTNESS = RAW / "m280-appworld-first-action-exactness-v1.json"
BODY_WEIGHTS = RAW / "m280-appworld-long-weight-v1.json"
COMBINED_WEIGHTS = RAW / "m280-appworld-long-heads-weight-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_receipt_hash(path: Path) -> dict:
    payload = _load(path)
    expected = payload.pop("receipt_self_sha256")
    assert _canonical_hash(payload) == expected
    return payload


def test_m280_long_sft_is_source_disjoint_and_sequence_exact_under_teacher_forcing() -> None:
    report = _load(SFT)
    assert report["kind"] == "localagent_public_agent_continuation_report"
    assert report["source"]["dataset"] == "appworld"
    assert report["source"]["revision"] == "appworld-0.2.0-data-0.2.0"
    assert report["rows"] == {"train": 24, "eval": 12}
    assert report["hyperparameters"]["steps"] == 256
    assert report["after"]["eval"]["assistant_token_accuracy"] > 0.96
    assert report["after"]["eval"]["assistant_sequence_accuracy"] == 0.75


def test_m280_long_body_free_run_and_head_combination_are_separate() -> None:
    runpython = _assert_receipt_hash(RUNPYTHON)
    combined = _assert_receipt_hash(COMBINED)
    assert runpython["configuration"]["replay_run_python"] is True
    assert runpython["summary"] == {
        "action_replayed": 0,
        "native_action_api_calls": 0,
        "native_api_calls": 0,
        "native_bootstrap_api_calls": 0,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert combined["configuration"]["selector_first"] is True
    assert combined["summary"] == {
        "action_replayed": 12,
        "native_action_api_calls": 12,
        "native_api_calls": 48,
        "native_bootstrap_api_calls": 36,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert combined["environment"]["native_runtime_executed"] is True
    assert combined["environment"]["environment_reset_per_task"] is True


def test_m280_first_action_hashes_and_weight_movement_keep_claim_boundary() -> None:
    exactness = _assert_receipt_hash(EXACTNESS)
    assert exactness["metrics"] == {
        "exact_code": 0,
        "exact_code_rate": 0.0,
        "predicted_code_available": 12,
        "rows": 12,
    }
    assert "first-action exactness only" in exactness["claim_boundary"]
    body = _load(BODY_WEIGHTS)
    combined = _load(COMBINED_WEIGHTS)
    for report in (body, combined):
        assert report["kind"] == "localagent_weight_transfer_analysis"
        assert report["compatibility"]["config_mismatches"] == {}
        assert report["compatibility"]["shape_mismatches"] == {}
        assert report["compatibility"]["tokenizer_sha256_equal"] is True
    assert body["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.035 < body["groups"]["embedding"]["relative_delta_l2"] < 0.037
    assert 0.013 < body["groups"]["attention_or_mixer"]["relative_delta_l2"] < 0.015
    assert 0.017 < body["groups"]["ffn"]["relative_delta_l2"] < 0.019
    assert 0.65 < combined["groups"]["action_heads"]["relative_delta_l2"] < 0.67
