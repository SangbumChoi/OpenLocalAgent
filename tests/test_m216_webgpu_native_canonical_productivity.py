from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m216-webgpu-native-canonical-productivity-v1.json"
APP = ROOT / "spaces/localagent-webgpu/app.js"


def _payload() -> dict:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == expected
    return payload


def test_m216_receipt_proves_native_canonical_schema_boundary() -> None:
    payload = _payload()
    assert payload["backend"] == "webgpu"
    assert payload["environment_executed"] is True
    assert payload["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert payload["capability"]["exact_actions"] == 3
    assert payload["capability"]["closed_loop_success"] == 0
    assert payload["capability"]["external_side_effects_executed"] is False
    assert [case["observed_tool"] for case in payload["capability"]["cases"]] == [
        "send_email",
        "open_url",
        "notion_write",
    ]


def test_m216_keeps_guard_claim_boundary_and_canonical_precedence() -> None:
    payload = _payload()
    contract = payload["canonical_contract"]
    assert contract["legacy_aliases"] == {
        "email_send": "send_email",
        "notion_create_page": "notion_write",
    }
    assert "not a learned-quality gain" in contract["interpretation"]
    assert "canonical schema-boundary" in payload["claim_boundary"]

    app = APP.read_text(encoding="utf-8")
    assert 'names.has("send_email") ? "send_email"' in app
    assert 'names.has("notion_write") ? "notion_write"' in app


def test_m216_performance_and_checkpoint_are_within_webgpu_budget() -> None:
    payload = _payload()
    assert payload["performance"]["tokens_per_second_p50"] > 100
    assert payload["performance"]["latency_ms_p50"] < 100
    assert payload["checkpoint"]["parameters"] < 100_000_000
    assert payload["deployment_bundle"]["parity_gate_passed"] is True
