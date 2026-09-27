"""Integrity checks for the targeted MCPMark filesystem native replay."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m489-mcpmark-filesystem-native-head-adaptation-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m489_binds_public_source_and_checkpoint_lineage() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mcpmark_filesystem_native_head_adaptation_receipt"
    assert payload["source"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["source"]["official_split_verified"] is False
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 1
    assert payload["success_rate"] == 1.0
    assert payload["training"]["continuation"]["rows"] == {"train": 8, "eval": 10}


def test_m489_native_verifier_passes_one_isolated_task() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["native"]["environment"]["mcp_server_executed"] is True
    assert payload["native"]["verifier_exit_code"] == 0
    assert payload["native"]["model_completed_task"] is False
    assert [turn["tool"] for turn in payload["native"]["turns"]] == ["directory_tree", "write_file"]
    assert payload["decision"]["promotion"].startswith("blocked_pending")
