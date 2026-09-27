import hashlib
import json
from pathlib import Path


def test_m607_current_policy_transfer_is_parent_bound_and_honest() -> None:
    path = Path("docs/paper/results/raw/m607-m585-policy-aligned-transfer-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["parent_checkpoint"]["sha256"].startswith("6553dc2b")
    assert payload["protocol"]["rows"] == {"train": 32, "eval": 40}
    assert payload["protocol"]["held_out_labels"] == ["agentnet"]
    assert payload["decision"]["warm_beats_random_all_surfaces"] is True
    assert payload["decision"]["warm_minus_random_after_pp"] == 62.895460797799174
    assert payload["decision"]["export_child_to_webgpu"] is False
    assert payload["decision"]["promote_toolace_to_executable_training_plan"] is False
    assert payload["weight_transfer"]["warm"]["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert "not an official score" in payload["claim_boundary"]
