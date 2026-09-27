"""Integrity checks for the native MCPMark Playwright replay."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m480-mcpmark-native-playwright-replay-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m480_binds_native_server_task_and_parent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mcpmark_native_playwright_replay_receipt"
    assert payload["source"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["source"]["official_split_verified"] is False
    assert payload["runtime"]["mcp_server_executed"] is True
    assert payload["runtime"]["verifiers_executed"] is True
    assert payload["parent"]["checkpoint"]["sha256"] == PARENT_SHA256


def test_m480_warm_child_reaches_mcp_tools_but_fails_verifier() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = payload["warm_child"]
    assert warm["summary"] == {"runtime_errors": 0, "tasks": 1, "verifier_failures": 1, "verifier_passes": 0}
    assert payload["comparison"]["warm_tool_calls"] == 3
    assert warm["task_result"]["turns"][0]["tool"] == "browser_navigate"
    assert warm["task_result"]["verifier_exit_code"] == 1
