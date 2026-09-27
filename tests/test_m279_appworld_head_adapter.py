"""Integrity checks for the AppWorld route/selector head-only repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/paper/results/raw"
ADAPTER = RAW / "m279-appworld-head-adapter-v1.json"
NATIVE = RAW / "m279-appworld-head-native-v1.json"
WEIGHTS = RAW / "m279-appworld-head-weight-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def test_m279_head_adapter_is_source_disjoint_and_routes_on_dev() -> None:
    report = json.loads(ADAPTER.read_text(encoding="utf-8"))
    assert report["kind"] == "localagent_public_agent_continuation_report"
    assert report["source"]["dataset"] == "appworld"
    assert report["source"]["revision"] == "appworld-0.2.0-data-0.2.0"
    assert report["rows"] == {"train": 24, "eval": 12}
    assert report["hyperparameters"]["head_steps"] == 256
    assert report["heads"]["before"]["route_accuracy"] == 0.0
    assert report["heads"]["before"]["selector_top1_accuracy"] == 0.0
    assert report["heads"]["after"]["route_accuracy"] == 1.0
    assert report["heads"]["after"]["selector_top1_accuracy"] == 1.0


def test_m279_native_receipt_replays_calls_but_verifiers_still_fail() -> None:
    receipt = json.loads(NATIVE.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert _canonical_hash(receipt) == expected
    assert receipt["configuration"]["selector_first"] is True
    assert receipt["configuration"]["schema_ground_appworld_api_step"] is True
    assert receipt["environment"] == {
        "environment_reset_per_task": True,
        "external_accounts": False,
        "native_runtime_executed": True,
        "screenshots": False,
        "state_side_effects": "isolated AppWorld task databases only",
    }
    assert receipt["summary"] == {
        "action_replayed": 9,
        "native_action_api_calls": 9,
        "native_api_calls": 48,
        "native_bootstrap_api_calls": 27,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert sum(task["action_replayed"] for task in receipt["tasks"]) == 9
    assert sum(task["evaluation"]["success"] for task in receipt["tasks"]) == 0
    assert "not an AppWorld leaderboard result" in receipt["claim_boundary"]


def test_m279_head_weight_audit_freezes_shared_body() -> None:
    report = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    assert report["kind"] == "localagent_weight_transfer_analysis"
    assert report["compatibility"]["config_mismatches"] == {}
    assert report["compatibility"]["shape_mismatches"] == {}
    assert report["compatibility"]["tokenizer_sha256_equal"] is True
    groups = report["groups"]
    assert groups["embedding"]["relative_delta_l2"] == 0.0
    assert groups["attention_or_mixer"]["relative_delta_l2"] == 0.0
    assert groups["ffn"]["relative_delta_l2"] == 0.0
    assert groups["normalization"]["relative_delta_l2"] == 0.0
    assert 0.65 < groups["action_heads"]["relative_delta_l2"] < 0.67
