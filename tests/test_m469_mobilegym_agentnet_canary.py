"""Integrity checks for the native MobileGym AgentNet canary."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m469-mobilegym-agentnet-warm-parent-canary-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m469_binds_mobilegym_revision_and_parent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mobilegym_agentnet_warm_parent_canary"
    assert payload["parent"]["checkpoint_sha256"] == PARENT_SHA256
    assert payload["protocol"]["task_count"] == 4
    assert payload["protocol"]["official_split_verified"] is True
    assert payload["protocol"]["native_receipt_eligible"] is False


def test_m469_agentnet_warm_does_not_improve_native_mobile_canary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["parent_success_rate"] == 0.0
    assert comparison["warm_success_rate"] == 0.0
    assert comparison["parent_progress_sum"] == 0.0
    assert comparison["warm_progress_sum"] == 0.0
    assert comparison["parent_tool_counts"] == {"mobile_input_text": 4}
    assert comparison["warm_tool_counts"] == {"mobile_input_text": 4}
    assert payload["decision"]["webgpu_export_allowed"] is False
