import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m710-workshop-gate-current-m679-v1.json")


def test_m710_gate_stays_fail_closed_with_agentnet_evidence() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m710_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:agentnet"].startswith("visual_runtime")
    assert "agentnet_visual_source" in payload["evidence"]
    assert "agentnet_selector_transfer" in payload["evidence"]
    assert len(payload["receipt_self_sha256"]) == 64
