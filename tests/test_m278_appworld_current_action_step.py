"""Integrity checks for the current native AppWorld action-step continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/paper/results/raw"
BASELINE = RAW / "m278-appworld-current-checkpoint-native-v1.json"
TRAINING = Path("/private/tmp/m278-appworld-action-current.json")
NATIVE = RAW / "m278-appworld-action-step-native-v1.json"
WEIGHTS = RAW / "m278-appworld-action-step-weight-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m278_current_appworld_baseline_is_native_and_zero_action() -> None:
    receipt = _load(BASELINE)
    expected = receipt.pop("receipt_self_sha256")
    assert _canonical_hash(receipt) == expected
    assert receipt["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert receipt["runner"]["contract_verification"]["passed"] == 1
    assert receipt["environment"] == {
        "environment_reset_per_task": True,
        "external_accounts": False,
        "native_runtime_executed": True,
        "screenshots": False,
        "state_side_effects": "isolated AppWorld task databases only",
    }
    assert receipt["summary"] == {
        "action_replayed": 0,
        "native_action_api_calls": 0,
        "native_api_calls": 0,
        "native_bootstrap_api_calls": 0,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 6,
    }
    assert "not an AppWorld leaderboard result" in receipt["claim_boundary"]


def test_m278_action_step_is_source_disjoint_but_not_sequence_exact() -> None:
    report = _load(TRAINING)
    assert report["kind"] == "localagent_public_agent_continuation_report"
    assert report["source"]["dataset"] == "appworld"
    assert report["source"]["revision"] == "appworld-0.2.0-data-0.2.0"
    assert report["rows"] == {"train": 24, "eval": 12}
    assert report["after"]["eval"]["assistant_token_accuracy"] > report["before"]["eval"][
        "assistant_token_accuracy"
    ]
    assert report["after"]["eval"]["assistant_sequence_accuracy"] == 0.0


def test_m278_action_step_native_replay_is_resettable_but_still_zero() -> None:
    receipt = _load(NATIVE)
    expected = receipt.pop("receipt_self_sha256")
    assert _canonical_hash(receipt) == expected
    assert receipt["configuration"]["action_translation"] == "appworld_api_step"
    assert receipt["configuration"]["schema_ground_appworld_api_step"] is True
    assert receipt["environment"]["native_runtime_executed"] is True
    assert receipt["environment"]["environment_reset_per_task"] is True
    assert receipt["summary"] == {
        "action_replayed": 0,
        "native_action_api_calls": 0,
        "native_api_calls": 0,
        "native_bootstrap_api_calls": 0,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert all(task["action_replayed"] is False for task in receipt["tasks"])
    assert "not an AppWorld leaderboard result" in receipt["claim_boundary"]


def test_m278_action_step_weight_audit_is_compatible_and_freezes_heads() -> None:
    report = _load(WEIGHTS)
    assert report["kind"] == "localagent_weight_transfer_analysis"
    assert report["compatibility"]["config_mismatches"] == {}
    assert report["compatibility"]["shape_mismatches"] == {}
    assert report["compatibility"]["tokenizer_sha256_equal"] is True
    groups = report["groups"]
    assert groups["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.004 < groups["embedding"]["relative_delta_l2"] < 0.005
    assert 0.002 < groups["attention_or_mixer"]["relative_delta_l2"] < 0.003
    assert 0.003 < groups["ffn"]["relative_delta_l2"] < 0.004
    assert groups["normalization"]["relative_delta_l2"] < 0.001
