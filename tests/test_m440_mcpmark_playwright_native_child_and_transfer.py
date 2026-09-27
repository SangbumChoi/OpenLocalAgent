"""Integrity checks for the current-child MCPMark Playwright diagnostic and transfer audit."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m440-mcpmark-playwright-native-child-and-transfer-v1.json"
)
CURRENT_CHILD_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m440_native_playwright_is_superseded_by_schema_corrected_receipt() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["checkpoint_sha256"] == CURRENT_CHILD_SHA256
    validity = payload["validity"]
    assert validity["status"] == "superseded"
    assert validity["superseded_by"] == "m445-mcpmark-playwright-schema-corrected-abi-guard-v1"
    assert validity["native_claim_usable"] is False
    assert validity["trajectory_transfer_usable"] is True
    native = payload["native_playwright"]
    assert native["mcp_server_executed"] is True
    assert native["official_split_verified"] is False
    assert native["verifier_passes"] == 0
    assert native["verifier_failures"] == payload["task_count"] == 4
    assert native["runtime_errors"] == 0
    assert all(task["verifier_exit_code"] == 1 for task in native["tasks"])


def test_m440_redacted_playwright_transfer_has_source_disjoint_control() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    transfer = payload["trajectory_transfer"]
    assert transfer["train_rows"] == 1
    assert transfer["source_disjoint_eval_rows"] == 2
    warm = transfer["warm"]
    random = transfer["random"]
    assert warm["backbone_init"] == "parent"
    assert random["backbone_init"] == "random"
    assert warm["after"]["token_accuracy"] > warm["before"]["token_accuracy"]
    assert transfer["warm_advantage_after_points"] > 30.0
    assert warm["after"]["sequence_accuracy"] == random["after"]["sequence_accuracy"] == 0.0
    assert warm["weight_transfer"]["embedding"]["relative_delta_l2"] < 0.01
    assert random["weight_transfer"]["embedding"]["relative_delta_l2"] > 1.0
