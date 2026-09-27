import hashlib
import json
from pathlib import Path


def test_m631_current_parent_ablation_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m631-m626-mcp-warm-random-transfer-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["parent_checkpoint"]["sha256"] == "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
    assert payload["decision"]["current_checkpoint_weight_gate"] is True
    assert payload["comparison"]["aggregate"]["warm_start_better_after"] is True
