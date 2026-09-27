"""Integrity checks for the WebGPU clarification/confirmation policy receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m231-webgpu-user-in-the-loop-clarification-policy-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m231_receipt_binds_user_in_the_loop_policy_states() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["policy_versions"] == {
        "side_effect": "side_effect_confirmation_v1",
        "interaction": "user_in_the_loop_v1",
    }
    assert [case["observed_status"] for case in payload["cases"]] == [
        "clarification_required",
        "confirmation_required",
        "allowed",
        "blocked",
    ]
    assert payload["cases"][0]["missing_arguments"] == ["recipient"]
    assert all(case["external_side_effects_executed"] is False for case in payload["cases"])
    assert payload["learned_model_changed"] is False
