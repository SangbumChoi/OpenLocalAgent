"""Integrity checks for the current-child BrowserGym run and gate refresh."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROWSERGYM = ROOT / "docs/paper/results/raw/m431-browsergym-native-child-full-v1.json"
GATE = ROOT / "docs/paper/results/raw/m432-workshop-gate-current-child-browsergym-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m431_is_complete_official_split_and_child_bound() -> None:
    payload = json.loads(BROWSERGYM.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "browsergym_miniwob"
    assert payload["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["source"]["browsergym_revision"] == "9e779f087de9a65668b6974d11f9ce9816026e96"
    assert payload["source"]["miniwob_revision"] == "7fd85d71a4b60325c6585396ec4f48377d049838"
    assert payload["environment"]["official_split_verified"] is True
    assert payload["environment"]["native_receipt_eligible"] is True
    assert payload["environment"]["task_count"] == payload["environment"]["expected_episodes"] == 240
    assert payload["environment"]["coordinate_fallback"] is False
    assert payload["result"]["success_rate"] == 0.0
    assert payload["result"]["successful_episodes"] == 0
    assert payload["result"]["steps"] == 2400
    assert payload["result"]["grounded_steps"] == 500
    assert payload["result"]["noop_actions"] == 1900


def test_m432_admits_mobile_and_browser_but_keeps_gate_closed() -> None:
    payload = json.loads(GATE.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["checks"]["native_mobilegym"] == "pass:official_split_verified"
    assert payload["checks"]["native_browsergym_miniwob"] == "pass:official_split_verified"
    assert payload["checks"]["weights_transfer_and_random_ablation"] == "pass"
    assert payload["checks"]["rl_preflight_current_child"] == "blocked:preflight_status_not_passed"
    assert "native:mobilegym" not in payload["blocking_requirements"]
    assert "native:browsergym_miniwob" not in payload["blocking_requirements"]
    assert "native:toolsandbox:official_split_not_verified" in payload["blocking_requirements"]
    assert "native:mcpmark:official_split_not_verified" in payload["blocking_requirements"]
    assert "training:rl_preflight:preflight_status_not_passed" in payload["blocking_requirements"]
