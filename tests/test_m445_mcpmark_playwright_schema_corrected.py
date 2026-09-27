"""Integrity checks for the schema-corrected MCPMark Playwright native diagnostic."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m445-mcpmark-playwright-schema-corrected-abi-guard-v1.json"
)
CURRENT_CHILD_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m445_receipt_binds_current_child_and_correct_schema_bridge() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mcpmark_playwright_schema_corrected_abi_guard"
    assert payload["checkpoint_sha256"] == CURRENT_CHILD_SHA256
    bridge = payload["schema_bridge"]
    assert bridge["mcp_python_sdk_key"] == "input_schema"
    assert bridge["accepted_fallback_key"] == "inputSchema"
    assert bridge["m440_superseded"] is True
    assert bridge["tool_count"] == 22


def test_m445_native_suites_are_real_but_not_official_and_not_promoted() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for key in ("native_playwright", "native_selector_warm"):
        native = payload[key]
        assert native["mcp_server_executed"] is True
        assert native["official_split_verified"] is False
        assert native["user_simulator_executed"] is False
        assert native["task_count"] == 4
        assert native["verifier_passes"] == 0
        assert native["verifier_failures"] == 4
        assert native["runtime_errors"] == 0
        assert all(result["verifier_exit_code"] == 1 for result in native["results"])
    assert payload["native_playwright"]["model_sha256"] == CURRENT_CHILD_SHA256
    assert payload["native_selector_warm"]["model_sha256"] != CURRENT_CHILD_SHA256


def test_m445_abi_guard_is_explicitly_non_learned_and_canary_is_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    guard = payload["abi_guard"]
    assert guard["not_learned_performance"] is True
    assert "browser_navigate" in guard["behavior"][0]
    assert "browser_snapshot" in guard["behavior"][1]
    assert "no candidate" in guard["behavior"][2]
    canary = guard["canary"]
    assert canary["sha256"]
    assert canary["bytes"] > 0


def test_m445_transfer_preserves_source_disjoint_control() -> None:
    transfer = json.loads(RECEIPT.read_text(encoding="utf-8"))["trajectory_transfer"]
    assert transfer["train_rows"] == 1
    assert transfer["source_disjoint_eval_rows"] == 2
    assert transfer["warm"]["after"]["token_accuracy"] > transfer["warm"]["before"]["token_accuracy"]
    assert transfer["random"]["backbone_init"] == "random"
    assert transfer["warm"]["after"]["sequence_accuracy"] == 0.0
