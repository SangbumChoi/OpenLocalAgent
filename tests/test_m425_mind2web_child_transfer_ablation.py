"""Integrity checks for the child-bound warm/random Mind2Web transfer ablation."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m425-mind2web-child-transfer-ablation-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m425_is_matched_and_child_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["parent_checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["dataset"]["parent_overlap"] == 0
    assert payload["dataset"]["typed_slot_overlap"] == 0
    assert payload["comparison"]["arm_contract"] == {
        "warm_backbone_init": "parent",
        "random_backbone_init": "random",
        "random_backbone_seed": 2028,
    }


def test_m425_supports_transfer_without_sequence_completion_claim() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["aggregate"]["warm_minus_random_after_pp"] > 70.0
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert comparison["warm"]["sequence_accuracy"] == 0.0
    assert comparison["random"]["sequence_accuracy"] == 0.0
    assert payload["weight_transfer_analysis"]["warm"]["compatibility"]["tokenizer_sha256_equal"] is True
