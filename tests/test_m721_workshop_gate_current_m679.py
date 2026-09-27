import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m721-workshop-gate-current-m679-v1.json"


def test_m721_gate_records_scaled_transfer_blocker() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m721_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert "scaled_transfer" in payload["blocked_requirements"]["artifacts:visual_webgpu_export"]
    assert "androidcontrol_structured_visual_scaling" in payload["evidence"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
