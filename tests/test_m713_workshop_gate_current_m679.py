import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m713-workshop-gate-current-m679-v1.json")


def test_m713_gate_binds_visual_pilot_and_stays_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m713_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:androidcontrol_visual"].startswith("visual_pilot")
    assert "androidcontrol_visual_bridge" in payload["evidence"]
    assert "androidcontrol_visual_action_pilot" in payload["evidence"]
    assert len(payload["receipt_self_sha256"]) == 64
