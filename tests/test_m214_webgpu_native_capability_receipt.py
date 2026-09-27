import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m214_receipt_is_native_webgpu_but_not_quality_pass() -> None:
    path = Path("docs/paper/results/raw/m214-webgpu-native-capability-current-bundle-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.pop("receipt_self_sha256")
    assert claimed == _canonical_sha256(receipt)
    assert receipt["backend"] == "webgpu"
    assert receipt["environment_executed"] is True
    assert receipt["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert receipt["capability"]["evaluated_cases"] == 3
    assert receipt["capability"]["exact_actions"] == 1
    assert receipt["capability"]["closed_loop_success"] == 0
    assert receipt["performance"]["tokens_per_second_p50"] > 100
    assert receipt["capability"]["external_side_effects_executed"] is False
