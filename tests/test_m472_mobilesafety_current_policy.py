"""Integrity checks for the current WebGPU MobileSafetyBench policy projection."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m472-mobilesafety-current-policy-v1.json")
TASK_SHA256 = "a44664bc7a575e69f6ac65c7bd07f1620dd177c49bf15a8ac029de5fa25540a4"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_m472_binds_current_policy_and_public_source() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mobilesafety_text_policy_projection"
    assert payload["source"]["dataset"] == "MobileSafetyBench"
    assert payload["source"]["revision"] == "bc5e0579626a280c4f551261abcb721442ff92ea"
    assert payload["source"]["tasks"]["sha256"] == TASK_SHA256
    assert payload["policy"]["native_execution"] is False
    assert payload["policy"]["external_side_effects"] is False


def test_m472_safety_boundary_is_stable_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["rows"] == 90
    assert summary["policy_status_counts"] == {
        "allowed": 23,
        "blocked": 22,
        "confirmation_required": 45,
    }
    assert summary["rows_with_prompt_injection_indicators"] == 1
    assert payload["qa_summary"]["policy_status_counts"] == {
        "blocked": 1,
        "confirmation_required": 2,
    }
    assert "official safety score" in payload["claim_boundary"]
