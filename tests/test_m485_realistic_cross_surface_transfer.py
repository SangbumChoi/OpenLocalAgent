"""Integrity checks for the balanced multi-surface continuation canary."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m485-realistic-cross-surface-transfer-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m485_binds_public_sources_parent_and_caps() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_realistic_cross_surface_transfer_receipt"
    assert payload["parent"]["sha256"] == PARENT_SHA256
    assert payload["training"]["rows"] == {"train": 59, "eval": 18}
    assert payload["training"]["cap_contract"]["purpose"] == "balanced_canary_not_final_training_run"
    assert payload["sources"]["agentnet"]["revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert payload["sources"]["mcpmark"]["official_split_verified"] is False


def test_m485_warm_beats_matched_random_but_exact_sequences_remain_zero() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["aggregate"]["warm_start_better_after"] is True
    assert payload["aggregate"]["warm_minus_random_after_pp"] > 40.0
    assert all(item["warm_start_better_after"] for item in payload["surfaces"].values())
    assert all(item["warm_start"]["after_token_accuracy"] >= 0.0 for item in payload["surfaces"].values())
    assert payload["claim_boundary"].startswith("Balanced public-train-only")
