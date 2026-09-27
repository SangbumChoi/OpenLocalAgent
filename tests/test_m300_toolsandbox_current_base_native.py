import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m300-toolsandbox-current-base-v1.json")


def test_m300_current_toolsandbox_base_receipt_is_fail_closed_and_complete() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == "toolsandbox"
    assert payload["environment_executed"] is True
    assert payload["verifier_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["user_simulator_executed"] is False
    assert payload["task_count"] == 129
    assert payload["success_count"] == 29
    assert payload["exceptions"] == 0
    assert set(payload["category_summary"]) >= {
        "NO_DISTRACTION_TOOLS",
        "INSUFFICIENT_INFORMATION",
        "MULTIPLE_TOOL_CALL",
        "MULTIPLE_USER_TURN",
        "CANONICALIZATION",
        "STATE_DEPENDENCY",
    }
    assert "not the official split" in payload["claim_boundary"]
