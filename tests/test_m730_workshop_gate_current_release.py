import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m730-workshop-gate-current-release-v1.json"


def test_m730_gate_keeps_toolsandbox_official_split_blocked() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m730_workshop_gate_current_release"
    assert payload["ready"] is False
    assert "interactive_0_of_3" in payload["blocked_requirements"]["native:toolsandbox"]
    assert "current_checkpoint_toolsandbox_native" in payload["evidence"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
