import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m360-current-webgpu-deploy-verification-v1.json"


def test_m360_current_bundle_is_checkpoint_and_manifest_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_webgpu_demo_deploy_verification"
    assert payload["verified"] is True
    assert payload["blockers"] == []
    assert payload["manifest"]["schema_version"] == 3
    assert payload["manifest"]["parity_gate_passed"] is True
    assert payload["manifest"]["sha256"] == (
        "f0c28409d75a03570c10aa4fa281e51ac1bc7b0c48608af1a7de3ea803ed94d7"
    )
    assert payload["bundle_identity_sha256"] == (
        "e0e19a8ad529a23e6bd703ed310eabe95f0a7b6b8ca7ad984c2d0c5ec8a296ed"
    )
    assert len(payload["artifacts"]) == 8
    assert all(item["verified"] for item in payload["artifacts"])


def test_m360_claim_boundary_does_not_overstate_webgpu_or_task_success() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    boundary = payload["claim_boundary"]
    assert "native WebGPU deployment" in boundary
    assert "every generated artifact" in boundary
