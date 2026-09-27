"""Integrity checks for the current-checkpoint AgentNet transfer receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m467-agentnet-current-warm-random-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m467_binds_agentnet_revision_and_parent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_agentnet_current_warm_random_transfer_receipt"
    assert payload["source"]["dataset"] == "xlangai/AgentNet"
    assert payload["source"]["revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert payload["parent_checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["protocol"]["split_contract"]["mode"] == (
        "source_local_parent_and_slot_disjoint"
    )
    assert payload["source"]["official_split_verified"] is False


def test_m467_warm_beats_random_but_exact_sequence_remains_zero() -> None:
    comparison = json.loads(RECEIPT.read_text(encoding="utf-8"))["comparison"]
    assert comparison["aggregate"]["warm_after_token_accuracy"] == 0.5252525252525253
    assert comparison["aggregate"]["random_after_token_accuracy"] == 0.0
    assert comparison["aggregate"]["warm_minus_random_after_pp"] > 52.5
    assert all(item["warm_start_better_after"] for item in comparison["surfaces"].values())

    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["warm"]["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert payload["random"]["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert payload["decision"]["export_child_to_webgpu"] is False
