"""Integrity checks for the current three-surface transfer control."""

from __future__ import annotations

import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m353-three-surface-transfer-current-v1.json")
CURRENT_CHECKPOINT = "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"


def _payload() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m353_binds_current_parent_and_matched_three_surface_rows() -> None:
    payload = _payload()
    assert payload["benchmark_id"] == "cross_surface_public_transfer"
    assert payload["checkpoint_sha256"] == CURRENT_CHECKPOINT
    assert payload["parent"]["sha256"] == CURRENT_CHECKPOINT
    assert payload["rows"] == {"train": 96, "eval": 24}
    assert {item["label"] for item in payload["train_sources"]} == {
        "androidcontrol",
        "agentnet",
        "mind2web",
    }
    assert {item["label"] for item in payload["eval_sources"]} == {
        "androidcontrol",
        "agentnet",
        "mind2web",
    }


def test_m353_warm_advantage_is_not_promoted_without_native_execution() -> None:
    payload = _payload()
    assert payload["native_execution"] is False
    assert payload["official_split_verified"] is False
    assert payload["deployment_decision"] == (
        "retain_parent_initialization_only_pending_native_replay"
    )
    assert payload["aggregate"]["warm_minus_random_after_pp"] > 50.0
    assert all(
        surface["warm_start_better_after"] for surface in payload["surfaces"].values()
    )
    assert payload["warm_weight_groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert payload["random_weight_groups"]["embedding"]["relative_delta_l2"] > 1.0
