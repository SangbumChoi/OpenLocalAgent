import hashlib
import json
from pathlib import Path


def test_m628_androidcontrol_webgpu_receipt_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m628-androidcontrol-webgpu-adoption-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["checkpoint"]["parameters"] == 10_524_544
    assert payload["webgpu_bundle"]["parity_gate_passed"] is True
    assert payload["native_webgpu"]["exact_actions"] == 3
    assert payload["native_webgpu"]["closed_loop_success"] == 0
    assert payload["adoption"]["public_model_published"] is False
