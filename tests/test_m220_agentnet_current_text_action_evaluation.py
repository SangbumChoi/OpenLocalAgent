"""Integrity checks for the current public AgentNet text-action evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m220-agentnet-current-text-action-evaluation-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m220_receipt_is_self_hashed_and_source_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["source"] == {
        "dataset": "xlangai/AgentNet",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/xlangai/AgentNet",
    }
    assert payload["projection"]["source_revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert payload["projection"]["action_rows"] == 133
    assert payload["projection"]["eval_parent_records"] == 8
    assert payload["projection"]["images_consumed"] is False
    assert payload["matched_control"]["checkpoint"]["bytes"] == 52319842


def test_m220_matched_control_is_complete_and_not_native() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = payload["warm_checkpoint"]["overall"]
    random = payload["matched_control"]["overall"]
    assert warm["tasks"] == random["tasks"] == 8
    assert warm["exact_trajectory_rate"] == random["exact_trajectory_rate"] == 0.0
    assert warm["first_action_type_rate"] > random["first_action_type_rate"]
    assert payload["decision"]["adopt_pretrained_backbone"] is False
    assert payload["decision"]["retain_as_diagnostic"] is True
    assert payload["weight_transfer"]["compatibility"]["shared_tensor_count"] == 51
    assert payload["weight_transfer"]["compatibility"]["tokenizer_sha256_equal"] is True
    assert "native desktop success" in payload["claim_boundary"]
