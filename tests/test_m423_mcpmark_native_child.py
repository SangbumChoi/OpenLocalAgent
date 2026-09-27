"""Integrity checks for the current-child MCPMark native filesystem receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m423-mcpmark-native-child-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m423_binds_child_and_executes_real_mcp_server() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["environment_executed"] is True
    assert payload["checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["mcp_runtime"]["mcp_server_executed"] is True
    assert payload["dataset"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"


def test_m423_reports_failure_without_official_claim() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["task_count"] == 1
    assert payload["success_rate"] == 0.0
    assert payload["rollout"]["model_completed_task"] is False
    assert payload["official_split_verified"] is False
