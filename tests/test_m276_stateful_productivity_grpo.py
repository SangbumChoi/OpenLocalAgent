"""Integrity checks for the bounded stateful-productivity GRPO continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "docs/paper/results/raw/m276-stateful-productivity-grpo-v1.json"
WEIGHTS = ROOT / "docs/paper/results/raw/m276-stateful-productivity-grpo-weight-v1.json"
PARENT_RUNTIME = ROOT / "docs/paper/results/raw/m276-stateful-productivity-runtime-parent-v1.json"
RL_RUNTIME = ROOT / "docs/paper/results/raw/m276-stateful-productivity-runtime-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m276_grpo_receipt_is_self_hashed_and_public_eval_is_isolated() -> None:
    payload = json.loads(TRAINING.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    assert _canonical_hash(payload) == expected
    assert payload["kind"] == "localagent_stateful_productivity_grpo_simulation"
    assert payload["source"]["public_benchmark_text_used"] is False
    assert payload["source"]["native_runtime_executed"] is False
    assert payload["configuration"]["deployment_heads_trainable"] is False
    assert payload["training"]["mean_reward_post"] > payload["training"]["mean_reward_pre"]
    assert payload["training"]["exact_match_accuracy_post"] == 0.0


def test_m276_weight_audit_confirms_tokenizer_and_frozen_heads() -> None:
    payload = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_weight_transfer_analysis"
    assert payload["compatibility"]["config_mismatches"] == {}
    assert payload["compatibility"]["shape_mismatches"] == {}
    assert payload["compatibility"]["tokenizer_sha256_equal"] is True
    assert payload["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.0 < payload["groups"]["embedding"]["relative_delta_l2"] < 0.03
    assert 0.0 < payload["groups"]["attention_or_mixer"]["relative_delta_l2"] < 0.02
    assert 0.0 < payload["groups"]["ffn"]["relative_delta_l2"] < 0.03


def test_m276_runtime_receipts_show_oracle_contract_and_real_model_gap() -> None:
    parent = json.loads(PARENT_RUNTIME.read_text(encoding="utf-8"))
    child = json.loads(RL_RUNTIME.read_text(encoding="utf-8"))
    for payload in (parent, child):
        expected = payload.pop("receipt_self_sha256")
        assert _canonical_hash(payload) == expected
        assert payload["oracle"]["task_complete_rate"] == 1.0
        assert payload["runtime"]["public_benchmark"] is False
    assert parent["model"]["task_complete_rate"] == 0.0
    assert child["model"]["task_complete_rate"] == 0.2
    assert child["model"]["abstention_exact"] == 1.0
    assert child["model"]["by_family"]["email"]["task_complete_rate"] == 0.0
    assert child["model"]["by_family"]["notion"]["task_complete_rate"] == 0.0
    assert child["model"]["by_family"]["browser"]["task_complete_rate"] == 0.0
    assert child["model"]["by_family"]["recovery"]["task_complete_rate"] == 0.0
