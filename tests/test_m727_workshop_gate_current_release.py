import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m727-workshop-gate-current-release-v1.json"


def test_m727_gate_records_current_native_browser_failure() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m727_workshop_gate_current_release"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:browsergym_current_checkpoint"].startswith(
        "bounded_native_text_slice_zero_of_eight"
    )
    assert "current_checkpoint_browsergym_native" in payload["evidence"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
