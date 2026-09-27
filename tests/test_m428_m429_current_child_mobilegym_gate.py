"""Integrity checks for the current-child official MobileGym run and gate refresh."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOBILEGYM = ROOT / "docs/paper/results/raw/m428-mobilegym-native-child-full-v1.json"
GATE = ROOT / "docs/paper/results/raw/m429-workshop-gate-current-child-mobilegym-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m428_is_complete_official_split_and_child_bound() -> None:
    payload = json.loads(MOBILEGYM.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mobilegym"
    assert payload["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert payload["environment"]["official_split_verified"] is True
    assert payload["environment"]["native_receipt_eligible"] is True
    assert payload["environment"]["task_count"] == payload["environment"]["official_test_task_count"] == 256
    assert payload["environment"]["full_official_test_split"] is True
    assert payload["environment"]["errors"] == []
    assert payload["result"]["passed_tasks"] == 1
    assert payload["result"]["failed_tasks"] == 255
    assert payload["result"]["success_rate"] == 1 / 256
    assert payload["environment"]["vision_used"] is False
    assert payload["comparison"]["child_minus_parent_success_rate_pp"] == 0.0


def test_m429_admits_mobilegym_but_keeps_publication_gate_closed() -> None:
    payload = json.loads(GATE.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["checks"]["native_mobilegym"] == "pass:official_split_verified"
    assert payload["checks"]["weights_transfer_and_random_ablation"] == "pass"
    assert payload["checks"]["rl_preflight_current_child"] == "blocked:preflight_status_not_passed"
    assert "native:mobilegym" not in payload["blocking_requirements"]
    assert "native:toolsandbox:official_split_not_verified" in payload["blocking_requirements"]
    assert "native:mcpmark:official_split_not_verified" in payload["blocking_requirements"]
    assert "training:rl_preflight:preflight_status_not_passed" in payload["blocking_requirements"]
