import json
from pathlib import Path


def test_m132_webgpu_m129_bundle_is_manifest_and_parity_verified() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m132-webgpu-m129-deploy-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["kind"] == "localagent_webgpu_demo_deploy_verification"
    assert receipt["verified"] is True
    assert receipt["deployment_bundle_present"] is True
    assert receipt["blockers"] == []
    assert receipt["manifest"]["schema_version"] >= 3
    assert receipt["manifest"]["parity_gate_passed"] is True
    assert receipt["bundle_identity_sha256"] == "c750c254140c47824b5cb82063b99ea0c22c1112606ee725bed23ae911251791"
    assert {entry["file"] for entry in receipt["artifacts"]} == {
        "action_model.fp16.onnx",
        "action_model.onnx",
        "dispatch_heads.json",
        "heads.json",
        "meta.json",
        "model.fp16.onnx",
        "model.onnx",
        "tokenizer.json",
    }
    assert all(entry["verified"] is True for entry in receipt["artifacts"])
