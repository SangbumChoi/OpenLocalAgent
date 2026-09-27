import hashlib
import json
from pathlib import Path


def test_m597_webgpu_receipt_is_canonical_and_checkpoint_bound() -> None:
    path = Path("docs/paper/results/raw/m597-m585-webgpu-capability-canonical-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["kind"] == "localagent_webgpu_native_capability_receipt"
    assert payload["backend"] == "webgpu"
    assert payload["environment_executed"] is True
    assert payload["capability"]["evaluated_cases"] == 3
    assert payload["capability"]["exact_actions"] == 3
    assert payload["performance"]["tokens_per_second_p50"] > 1000
