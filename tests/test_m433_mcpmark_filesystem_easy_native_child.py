"""Integrity checks for the current-child MCPMark filesystem easy diagnostic."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m433-mcpmark-filesystem-easy-native-child-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m433_binds_child_and_pinned_mcpmark_filesystem_runtime() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["dataset"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["environment"]["mcp_server_executed"] is True
    assert payload["environment"]["task_verifiers_executed"] is True
    assert payload["environment"]["official_split_verified"] is False


def test_m433_reports_all_easy_tasks_failed_without_official_claim() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["dataset"]["task_count"] == 10
    assert payload["summary"] == {
        "tasks": 10,
        "verifier_passes": 0,
        "verifier_failures": 10,
        "runtime_errors": 0,
        "changed_workspaces": 0,
    }
    assert len(payload["results"]) == 10
    assert all(item["verifier_exit_code"] == 1 for item in payload["results"])
    assert all(item["changed"] is False for item in payload["results"])
    assert "official MCPMark split" in payload["claim_boundary"]
