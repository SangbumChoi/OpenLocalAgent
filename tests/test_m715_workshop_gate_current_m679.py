import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m715-workshop-gate-current-m679-v1.json")


def test_m715_binds_structured_visual_evidence_and_stays_closed() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m715_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:androidcontrol_structured_visual"].startswith("structured")
    assert "androidcontrol_structured_visual_pilot" in payload["evidence"]
    assert len(payload["receipt_self_sha256"]) == 64
