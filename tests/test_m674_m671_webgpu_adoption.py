import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m674_binds_native_webgpu_to_local_release() -> None:
    path = ROOT / "docs/paper/results/raw/m674-m671-webgpu-adoption-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["checkpoint"]["sha256"].startswith("b5576dc8")
    assert payload["adoption"]["local_webgpu_adopted"] is True
    assert payload["adoption"]["public_model_published"] is False
    assert payload["native_webgpu"]["environment_executed"] is True
    assert payload["native_webgpu"]["evaluated_cases"] == 3
    assert payload["native_webgpu"]["exact_actions"] == 3
    assert payload["native_webgpu"]["external_side_effects_executed"] is False
    assert payload["native_webgpu"]["closed_loop_success"] == 0
