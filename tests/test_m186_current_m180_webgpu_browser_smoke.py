from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path(__file__).resolve().parents[1] / "docs/paper/results/raw/m186-current-m180-webgpu-browser-smoke-v1.json"


def test_m186_browser_smoke_is_hash_bound_and_provider_explicit() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert claimed == expected
    runtime = payload["runtime"]
    assert runtime["model_ready"] is True
    assert runtime["reported_provider"] == "webgpu"
    assert runtime["runtime_errors"] == 0
    assert runtime["bundle_identity_sha256"]


def test_m186_realistic_cases_fail_closed_and_have_no_side_effects() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 2
    assert all(case["success"] is False for case in payload["cases"])
    assert payload["security"]["external_accounts_used"] is False
    assert payload["security"]["external_side_effects"] is False
    assert payload["decision"] == "browser_execution_verified_quality_gate_failed"
    assert "not proof of 100–300 tokens/s" in payload["claim_boundary"]
