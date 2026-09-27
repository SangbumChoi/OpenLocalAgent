from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m252-tau2-mock-native-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def test_tau2_mock_native_receipt_is_resettable_and_fail_closed() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert _canonical_hash(receipt) == expected
    assert receipt["source"]["dataset"] == "tau2-bench"
    assert receipt["source"]["domain"] == "mock"
    assert receipt["source"]["license"] == "MIT"
    assert receipt["source"]["task_count"] == 10
    assert receipt["runner"]["source_revision"] == "363133ada1936491fb5bcec33cd62c3518a99f65"
    assert receipt["environment"] == {
        "external_accounts": False,
        "external_services": False,
        "native_runtime_executed": True,
        "reset_per_task": True,
        "screenshots": False,
    }
    assert receipt["contract_verification"]["passed"] is True
    assert receipt["summary"] == {
        "bounded_native_success_rate": 0.0,
        "bounded_native_successes": 0,
        "first_action_exact": 0,
        "model_tool_calls": 0,
        "tasks": 10,
    }
    assert all(task["model_tool_calls"] == 0 for task in receipt["tasks"])
    assert "not a tau2 leaderboard" in receipt["claim_boundary"]


def test_tau2_mock_receipt_does_not_retain_task_text_or_tool_outputs() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    assert "Important Meeting" not in serialized
    assert "Create a new task" not in serialized
    assert "task_id='task_2'" not in serialized
    for task in receipt["tasks"]:
        assert set(task["instruction"]) == {"bytes", "sha256"}
        assert set(task["model_output"]) == {"bytes", "sha256"}
