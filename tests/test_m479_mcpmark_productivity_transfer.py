"""Integrity checks for the source-bound MCPMark productivity transfer receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m479-mcpmark-productivity-transfer-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m479_binds_public_revision_and_parent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mcpmark_productivity_transfer_receipt"
    assert payload["source"]["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert payload["source"]["revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert payload["parent"]["sha256"] == PARENT_SHA256
    assert payload["training"]["rows"] == {"train": 2, "eval": 2}


def test_m479_warm_start_wins_but_native_gate_is_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["transfer_decision"] == "warm_start_dominates_matched_random_on_all_surfaces"
    assert payload["aggregate"]["warm_minus_random_after_pp"] > 30.0
    assert all(item["warm_start_better_after"] for item in payload["surfaces"].values())
    assert payload["native_replay"]["status"] == "blocked_by_missing_network_dependency"
    assert payload["native_replay"]["environment"]["official_split_verified"] is False
