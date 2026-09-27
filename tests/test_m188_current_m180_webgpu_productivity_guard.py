from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path(__file__).resolve().parents[1] / "docs/paper/results/raw/m188-current-m180-webgpu-productivity-guard-v1.json"


def test_m188_receipt_is_hash_bound_and_current_checkpoint_pinned() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert claimed == expected
    assert payload["checkpoint"]["parameters"] < 100_000_000
    assert payload["runtime"]["reported_provider"] == "webgpu"
    assert payload["runtime"]["model_ready"] is True


def test_m188_side_effect_guards_pass_and_abstention_failure_remains_visible() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    email, notion, generic = payload["cases"]
    assert email["tool"] == "send_email"
    assert email["schema_valid"] is True
    assert email["external_side_effect"] is False
    assert notion["tool"] == "notion_write"
    assert notion["arguments"] == {"content": "search result"}
    assert notion["schema_valid"] is True
    assert generic["schema_valid"] is False
    assert generic["intent_guard_passed"] is False
    assert payload["decision"] == "guarded_productivity_routing_passes_but_learned_quality_gate_fails"
