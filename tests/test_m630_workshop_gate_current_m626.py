import hashlib
import json
from pathlib import Path


def test_m630_gate_is_current_and_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m630-workshop-gate-current-m626-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["decision"]["ready"] is False
    assert payload["decision"]["native_mobile_evidence_current"] is True
    assert payload["decision"]["webgpu_evidence_current"] is True
