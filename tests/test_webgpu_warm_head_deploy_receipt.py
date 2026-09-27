"""Regression checks for the current warm-head WebGPU export receipt."""

import json
from pathlib import Path


def test_warm_head_webgpu_bundle_is_hash_and_parity_bound() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipt = json.loads((root / "m109-webgpu-warm-head-deploy-v1.json").read_text())
    assert receipt["kind"] == "localagent_webgpu_demo_deploy_verification"
    assert receipt["verified"] is True
    assert receipt["manifest"]["parity_gate_passed"] is True
    assert receipt["checkpoint"]["sha256"] == (
        "d81771f61a8391a1973c84e396b8c79e1dd94a8b28bf3e41f90b585ba1d57486"
    )
    assert receipt["checkpoint"]["parameters"] == 10524544
    assert receipt["export"]["parity_hard_gate"] is True
    artifact_names = {artifact["file"] for artifact in receipt["artifacts"]}
    assert {"action_model.fp16.onnx", "dispatch_heads.json", "tokenizer.json"} <= artifact_names
    assert "native WebGPU deployment" in receipt["claim_boundary"]
