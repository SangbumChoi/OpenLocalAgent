import hashlib
import json
from pathlib import Path


def test_m625_adoption_receipt_is_checkpoint_bound() -> None:
    path = Path("docs/paper/results/raw/m625-mcp-trajectory-webgpu-adoption-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["kind"] == "localagent_m625_mcp_trajectory_webgpu_adoption"
    assert payload["checkpoint"]["parameters"] < 100_000_000
    assert payload["webgpu_bundle"]["deployment_verified"] is True
    assert payload["webgpu_bundle"]["parity_gate_passed"] is True
    assert payload["native_webgpu"]["environment_executed"] is True
    assert payload["native_webgpu"]["external_side_effects_executed"] is False
    assert payload["adoption"]["public_model_published"] is False
