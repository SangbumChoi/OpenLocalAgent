import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m717-workshop-gate-current-m679-v1.json")


def test_m717_binds_visual_export_and_stays_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m717_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["artifacts:visual_webgpu_export"].startswith("cpu_onnx")
    assert "structured_visual_onnx_parity" in payload["evidence"]
    assert len(payload["receipt_self_sha256"]) == 64
