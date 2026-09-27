"""Integrity checks for the current Mind2Web browser-head transfer control."""

from __future__ import annotations

import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m356-current-mind2web-browser-head-transfer-v1.json")
CURRENT_CHECKPOINT = "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"


def _payload() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m356_binds_current_parent_and_public_train_only_split() -> None:
    payload = _payload()
    assert payload["checkpoint"]["parent"]["sha256"] == CURRENT_CHECKPOINT
    assert payload["source"]["revision"] == "17ece8eb89862368edc0cc806acee6fca5163474"
    assert payload["source"]["upstream_split"] == "train"
    assert payload["source"]["train"]["decisions"] == 219
    assert payload["source"]["eval"]["decisions"] == 63
    assert payload["source"]["split_audit"] == {
        "parent_record_disjoint": True,
        "typed_slot_disjoint": True,
        "official_test_consumed": False,
        "screenshots_consumed": False,
    }


def test_m356_matched_control_is_not_promoted_without_native_replay() -> None:
    payload = _payload()
    assert payload["native_execution"] is False
    assert payload["decision"]["promote_child_to_deployment"] is False
    assert payload["decision"]["adopt_browser_heads"] is False
    assert payload["warm"]["after"]["selector_top1"] == payload["matched_random"]["after"]["selector_top1"]
    assert payload["warm"]["selector_relative_delta_l2"] < payload["matched_random"]["selector_relative_delta_l2"]
    assert payload["decision"]["native_gate_status"] == "unchanged_blocked"
