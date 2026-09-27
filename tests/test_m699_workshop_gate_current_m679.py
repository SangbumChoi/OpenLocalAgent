import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m699-workshop-gate-current-m679-v1.json")


def test_m699_is_fail_closed_and_current_bound() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m699_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:mcpmark"] == "native_verifier_zero"
    assert payload["blocked_requirements"]["native:appworld"] == "native_completion_zero"
    assert payload["current_checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m699_records_latest_native_receipts() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert "mcpmark_native_filesystem_standard" in payload["evidence"]
    assert "mcpmark_trajectory_continuation_native" in payload["evidence"]
    assert "native:mcpmark" not in payload["passed_requirements"]
