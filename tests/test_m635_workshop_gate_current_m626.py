import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m635-workshop-gate-current-m626-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m635_gate_records_current_toolsandbox_without_faking_readiness() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["decision"]["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["decision"]["native_toolsandbox_evidence_current"] is True
    assert any(
        item["requirement"] == "native:toolsandbox"
        and "official_split_not_verified" in item["blockers"]
        for item in payload["decision"]["blocking_requirements"]
    )
