import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m646-workshop-gate-current-m626-grounding-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m646_gate_attaches_grounding_without_faking_readiness() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["current_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["decision"]["ready"] is False
    assert payload["decision"]["grounding_diagnostic_changes_readiness"] is False
    assert payload["grounding_summary"] == {
        "semantic_success_rate_baseline": 0.25,
        "semantic_success_rate_child": 0.25,
        "coordinate_success_rate_baseline": 0.5,
        "coordinate_success_rate_child": 0.5,
        "semantic_child_success_delta": 0.0,
        "coordinate_child_success_delta": 0.0,
    }
