"""Integrity checks for the native BrowserGym AgentNet canary."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m468-browsergym-agentnet-warm-parent-canary-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m468_binds_native_revisions_and_parent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_browsergym_agentnet_warm_parent_canary"
    assert payload["parent"]["checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["protocol"]["task_count"] == 4
    assert payload["protocol"]["official_split_verified"] is False


def test_m468_agentnet_warm_does_not_improve_native_browser_canary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["parent_success_rate"] == 0.0
    assert comparison["warm_success_rate"] == 0.0
    assert comparison["parent_grounded_steps"] == 0
    assert comparison["warm_grounded_steps"] == 0
    assert comparison["parent_noop_actions"] == 40
    assert comparison["warm_noop_actions"] == 40
    assert payload["decision"]["webgpu_export_allowed"] is False
