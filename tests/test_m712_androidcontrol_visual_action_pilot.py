import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m712-androidcontrol-visual-action-pilot-v1.json")


def test_m712_binds_parent_disjoint_visual_action_pilot() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m712_androidcontrol_visual_action_pilot"
    assert payload["model"]["parameters"] < 100_000_000
    assert payload["source"]["screenshot_bytes_consumed"] is True
    assert payload["split"]["parent_disjoint"] is True
    assert payload["split"]["train_samples"] > 0
    assert payload["split"]["eval_samples"] > 0
    assert payload["warm"]["training"]["vision_update_l2"] > 0
    assert payload["random"]["training"]["vision_update_l2"] > 0


def test_m712_keeps_native_and_webgpu_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert "official AndroidControl score" in payload["claim_boundary"]
    assert payload["weight_analysis"]["adoption_decision"].startswith("retain_warm")
