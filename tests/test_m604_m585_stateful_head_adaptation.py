import hashlib
import json
from pathlib import Path


def test_m604_stateful_head_adaptation_is_current_and_backbone_frozen() -> None:
    path = Path("docs/paper/results/raw/m604-m585-stateful-head-adaptation-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["parent_checkpoint"]["sha256"].startswith("6553dc2b")
    assert payload["protocol"]["backbone_frozen"] is True
    assert payload["warm_arm"]["schema_valid_rate"] == 1.0
    assert payload["warm_arm"]["by_family"]["email"]["task_complete"] == 1
    assert payload["warm_arm"]["by_family"]["recovery"]["task_complete"] == 0
    assert payload["head_transition"]["backbone_relative_l2"] == 0.0
