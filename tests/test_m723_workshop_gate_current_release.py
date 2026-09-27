import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m723-workshop-gate-current-release-v1.json"


def test_m723_current_release_gate_is_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m723_workshop_gate_current_release"
    assert payload["ready"] is False
    assert "current_checkpoint_structured_visual" in payload["evidence"]
    assert "pointer_regression" in payload["blocked_requirements"]["artifacts:visual_webgpu_export"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
