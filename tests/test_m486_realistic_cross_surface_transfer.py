"""Integrity checks for the 16-step matched realistic-agent canary."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m486-realistic-cross-surface-transfer-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_m486_binds_parent_sources_and_16_step_caps() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_realistic_cross_surface_transfer_receipt"
    assert payload["parent"]["sha256"] == PARENT_SHA256
    assert payload["training"]["rows"] == {"train": 59, "eval": 18}
    assert payload["training"]["cap_contract"]["steps"] == 16
    assert payload["training"]["cap_contract"]["purpose"] == "matched_canary_not_final_training_run"
    assert payload["sources"]["mcpmark"]["official_split_verified"] is False


def test_m486_warm_dominates_random_but_exact_sequences_are_zero() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["aggregate"]["warm_start_better_after"] is True
    assert payload["aggregate"]["warm_minus_random_after_pp"] > 40.0
    assert all(item["warm_start_better_after"] for item in payload["surfaces"].values())
    assert payload["aggregate"]["random_after_token_accuracy"] == 0.0
    assert all(item["warm_start"]["after_token_accuracy"] >= 0.0 for item in payload["surfaces"].values())
    assert payload["claim_boundary"].startswith("Matched public-train-only")
