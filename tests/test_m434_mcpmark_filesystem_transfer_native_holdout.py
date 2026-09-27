"""Integrity checks for the MCPMark state/argument transfer intervention."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m434-mcpmark-filesystem-transfer-native-holdout-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m434_binds_source_disjoint_transfer_and_current_child() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["dataset"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["dataset"]["train_task_families"] == ["file_context", "file_property"]
    assert payload["dataset"]["eval_task_families"] == [
        "folder_structure",
        "legal_document",
        "papers",
        "student_database",
    ]
    assert payload["training"]["source_disjoint"] is True
    assert payload["checkpoint"]["parent_sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )


def test_m434_teacher_forced_gain_does_not_overrule_native_failure() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    transfer = payload["teacher_forced_transfer"]
    native = payload["native_holdout"]
    assert transfer["token_accuracy_after"] > transfer["token_accuracy_before"]
    assert transfer["sequence_accuracy_after"] == 0.0
    assert native["verifier_passes"] == 0
    assert native["verifier_failures"] == native["tasks"] == 5
    assert native["changed_workspaces"] == 0
    assert payload["matched_random_control"]["status"] == "not_available"
    assert "rejected for promotion" in payload["claim_boundary"]
