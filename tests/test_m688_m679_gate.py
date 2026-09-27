import hashlib
import json
from pathlib import Path


def test_m688_gate_records_toolsandbox_protocol_blocker() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m688-workshop-gate-current-m679-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert expected == actual
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:toolsandbox"] == "official_split_not_verified"
    assert "weights:transfer_and_no_transfer_ablation" in payload["passed_requirements"]
