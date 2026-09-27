"""Integrity checks for the larger MCPMark cross-surface transfer ablation."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m482-mcpmark-cross-surface-transfer-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m482_binds_public_sources_and_parent_disjoint_rows() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mcpmark_cross_surface_transfer_receipt"
    assert payload["parent"]["sha256"] == PARENT_SHA256
    assert payload["training"]["rows"] == {"train": 11, "eval": 12}
    assert payload["training"]["split_contract"]["mode"] == "source_local_parent_and_slot_disjoint"
    assert payload["sources"]["trajectory_log"]["revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert payload["sources"]["mcpmark"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"


def test_m482_warm_wins_each_held_out_surface() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["transfer_decision"] == "warm_start_dominates_matched_random_on_all_surfaces"
    assert payload["aggregate"]["warm_minus_random_after_pp"] > 35.0
    assert all(item["warm_start_better_after"] for item in payload["surfaces"].values())
    assert payload["surfaces"]["mcpmark_filesystem"]["rows"] == 10
