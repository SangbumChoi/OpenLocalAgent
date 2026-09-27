import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m711-androidcontrol-visual-bridge-smoke-v1.json")


def test_m711_binds_public_screenshot_to_trainable_bridge() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m711_androidcontrol_visual_bridge_smoke"
    assert payload["model"]["parameters"] < 100_000_000
    assert payload["pipeline_boundary"]["screenshot_bytes_consumed"] is True
    assert payload["pipeline_boundary"]["vision_bridge_forward"] is True
    assert payload["pipeline_boundary"]["vision_bridge_update"] is True
    assert payload["training"]["vision_grad_norm"] > 0
    assert payload["training"]["vision_parameter_update_l2"] > 0


def test_m711_does_not_claim_deployment_or_quality() -> None:
    payload = json.loads(RECEIPT.read_text())
    boundary = payload["pipeline_boundary"]
    assert boundary["native_emulator_executed"] is False
    assert boundary["webgpu_exported"] is False
    assert boundary["quality_training_admitted"] is False
