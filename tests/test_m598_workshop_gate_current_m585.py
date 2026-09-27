import hashlib
import json
from pathlib import Path


def test_m598_current_gate_is_fail_closed_and_records_passed_evidence() -> None:
    path = Path("docs/paper/results/raw/m598-workshop-gate-current-m585-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"].startswith("6553dc2b")
    assert "native:browsergym_miniwob" in payload["passed_requirements"]
    assert "webgpu:native_capability_and_latency" in payload["passed_requirements"]
    assert "native:androidworld" in payload["blocked_requirements"]
    assert "artifacts:public_model_demo_manifest" in payload["blocked_requirements"]
