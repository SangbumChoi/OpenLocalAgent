"""Integrity checks for the structured-action safety regression receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m230-webgpu-structured-action-safety-fix-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m230_receipt_binds_structured_action_safety_fix() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["policy_version"] == "side_effect_confirmation_v1"
    assert "tool field" in payload["root_cause"]
    assert [case["observed_status"] for case in payload["cases"]] == [
        "confirmation_required",
        "confirmation_required",
        "allowed",
        "blocked",
    ]
    assert all(case["external_side_effects_executed"] is False for case in payload["cases"])
    assert payload["learned_model_changed"] is False
