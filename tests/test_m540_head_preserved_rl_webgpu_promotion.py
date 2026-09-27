import hashlib
import json
from pathlib import Path


def test_m540_head_preserved_rl_candidate_is_exportable_and_bounded() -> None:
    path = Path("docs/paper/results/raw/m540-head-preserved-rl-webgpu-promotion-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    raw = payload["comparison"]["raw_preflight_child"]
    assert raw["exportable_webgpu"] is False
    assert raw["structured_heads_available"] is False
    child = payload["comparison"]["head_preserved_child"]
    assert child["exportable_webgpu"] is True
    assert len(child["deployment_heads_preserved"]) == 5
    assert payload["local_trajectory"]["pass_at_1"] == 1.0
    assert payload["webgpu"]["exact_actions"] == payload["webgpu"]["evaluated_cases"] == 3
    assert payload["rl"]["body_movement_relative_l2"]["action_heads"] == 0.0
