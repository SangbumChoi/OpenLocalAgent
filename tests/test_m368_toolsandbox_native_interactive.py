from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m368-toolsandbox-native-current-interactive-v1.json"
CHECKPOINT_SHA256 = "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m368_toolsandbox_interactive_receipt_is_current_and_self_hashed() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["benchmark_id"] == "toolsandbox"
    assert receipt["source_revision"] == "165848b9a78cead7ca7fe7c89c688b58e6501219"
    assert receipt["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert receipt["environment_executed"] is True
    assert receipt["verifier_executed"] is True
    assert receipt["external_api_called"] is False
    assert receipt["protocol"] == "bounded_multi_step_scripted_user"
    assert receipt["max_agent_turns"] == 4
    assert receipt["task_count"] == 3
    assert receipt["success_count"] == 0
    assert receipt["success_rate"] == 0.0
    assert receipt["official_split_verified"] is False
    assert receipt["user_simulator_executed"] is False
    assert receipt["post_tool_response_policy"] == "model_continuation_bounded_no_deterministic_template"
    assert {item["scenario"] for item in receipt["scenarios"]} == {
        "cellular_off",
        "wifi_off",
        "send_message_with_phone_number_and_content",
    }


def test_m368_preserves_the_python_compatibility_boundary() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    compatibility = receipt["runtime_compatibility"]
    assert compatibility["polars_version"] == "0.20.31"
    assert "ccy==1.3.1" in compatibility["source_dependency_note"]
    assert "ccy==1.4.0" in compatibility["source_dependency_note"]
    assert "official-split requirement" in receipt["claim_boundary"]
