import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m725-workshop-gate-current-release-v1.json"


def test_m725_gate_keeps_native_success_blocked_after_browser_probe() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m725_workshop_gate_current_release"
    assert payload["ready"] is False
    assert "current_checkpoint_visual_webgpu_probe" in payload["evidence"]
    assert "native_mobile_verifier" in payload["blocked_requirements"]["artifacts:visual_webgpu_export"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
