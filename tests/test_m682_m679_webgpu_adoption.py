import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m682_binds_current_m679_webgpu_bundle_and_keeps_public_gate_closed() -> None:
    path = ROOT / "docs/paper/results/raw/m682-m679-webgpu-adoption-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["kind"] == "localagent_m682_m679_webgpu_adoption"
    assert payload["checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["native_webgpu"]["environment_executed"] is True
    assert payload["native_webgpu"]["exact_actions"] == 3
    assert payload["native_webgpu"]["external_side_effects_executed"] is False
    assert payload["adoption"]["public_model_published"] is False
