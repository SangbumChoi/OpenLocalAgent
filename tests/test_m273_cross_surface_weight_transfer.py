"""Integrity checks for the public cross-surface transfer receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m273-cross-surface-public-weight-transfer-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_m273_receipt_is_self_hashed_and_binds_public_splits() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert "google-research/google-research" in payload["sources"]["androidcontrol"]["original_url"]
    assert (
        payload["sources"]["agentnet"]["dataset_url"]
        == "https://huggingface.co/datasets/xlangai/AgentNet"
    )
    assert payload["protocol"]["train_rows_per_source"] == 512
    assert payload["protocol"]["eval_rows_per_source"] == 32
    assert payload["protocol"]["tokenizer_sha256_equal"] is True
    assert (
        payload["teacher_forced"]["warm_start"]["eval_after_token_accuracy"]
        > payload["teacher_forced"]["warm_start"]["eval_before_token_accuracy"]
    )
    assert payload["teacher_forced"]["random_backbone_control"]["eval_after_token_accuracy"] < 0.01
    assert payload["action_level_agentnet_control"]["warm_start"]["success_rate"] == 0.0
    assert "not official" in payload["claim_boundary"]
