import hashlib
import json
from pathlib import Path


def test_m627_gate_is_self_consistent_and_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m627-workshop-gate-current-m624-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["decision"]["ready"] is False
    assert payload["decision"]["webgpu_and_weight_evidence_current"] is True
    assert len(payload["decision"]["blocking_requirements"]) == 13
