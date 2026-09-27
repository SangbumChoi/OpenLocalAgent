"""Integrity checks for the matched AgentNet public continuation experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m221-agentnet-public-continuation-transfer-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m221_receipt_is_self_hashed_and_parent_disjoint() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["dataset"]["revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert payload["dataset"]["parent_disjoint"] is True
    assert payload["dataset"]["train"]["rows"] == 513
    assert payload["dataset"]["eval"]["rows"] == 133
    assert payload["dataset"]["eval"]["parent_records"] == 8
    assert payload["dataset"]["images_consumed"] is False


def test_m221_warm_transfer_beats_random_language_and_route_but_not_actions() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = payload["warm"]
    random = payload["random_control"]
    assert warm["after"]["eval_token_accuracy"] > random["after"]["eval_token_accuracy"]
    assert warm["after"]["eval_route_accuracy"] > random["after"]["eval_route_accuracy"]
    assert warm["action_evaluation"]["first_action_type_rate"] > random["action_evaluation"]["first_action_type_rate"]
    assert warm["action_evaluation"]["exact_trajectory_rate"] == 0.0
    assert random["action_evaluation"]["exact_trajectory_rate"] == 0.0
    assert warm["weight_audit"]["shared_tensor_count"] == 51
    assert random["weight_audit"]["shared_tensor_count"] == 51
    assert warm["weight_audit"]["tokenizer_sha256_equal"] is True
    assert random["weight_audit"]["tokenizer_sha256_equal"] is True
    assert "do_not_promote" in payload["comparison"]["decision"]
