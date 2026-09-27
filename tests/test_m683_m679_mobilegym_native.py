import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m683_mobilegym_binds_current_m679_and_rejects_promotion() -> None:
    path = ROOT / "docs/paper/results/raw/m683-m679-mobilegym-native-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["official_split_verified"] is True
    assert payload["task_count"] == 256
    assert payload["result"]["passed_tasks"] == 1
    assert payload["decision"]["native_mobile_promotion"] is False
    assert payload["protocol"]["vision_used"] is False
