import hashlib
import json
from pathlib import Path


def test_m551_gate_is_fail_closed_and_current_checkpoint_bound() -> None:
    path = Path("docs/paper/results/raw/m551-workshop-gate-current-m546-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["parameters"] < 100_000_000
    assert payload["checks"]["native_mobilegym"] == "pass"
    assert payload["checks"]["native_browsergym_miniwob"] == "pass"
    assert payload["checks"]["webgpu_native_capability_and_latency"] == "pass"
    assert payload["checks"]["training_rl_preflight"] == "blocked"
    assert payload["checks"]["artifacts_public_model_demo_manifest"] == "blocked"
