import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m716-structured-visual-onnx-parity-v1.json")


def test_m716_binds_trained_visual_sidecar_and_cpu_parity() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m716_structured_visual_onnx_parity"
    assert payload["parity"]["passed"] is True
    assert payload["deployment_boundary"]["onnx_cpu_runtime"] is True
    assert payload["deployment_boundary"]["onnx_webgpu_runtime"] is False
    assert payload["abi"]["inputs"]["images"] == [1, 3, 96, 96]
    assert payload["abi"]["outputs"]["action_logits"] == [1, 7]


def test_m716_does_not_claim_browser_or_native_success() -> None:
    payload = json.loads(RECEIPT.read_text())
    boundary = payload["deployment_boundary"]
    assert boundary["browser_demo_bound"] is False
    assert boundary["native_mobile_verifier"] is False
