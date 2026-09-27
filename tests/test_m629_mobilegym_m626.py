import hashlib
import json
from pathlib import Path


def test_m629_mobilegym_receipt_is_self_consistent_and_native() -> None:
    path = Path("docs/paper/results/raw/m629-androidcontrol-child-mobilegym-native-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["protocol"]["official_split_verified"] is True
    assert payload["protocol"]["task_count"] == 256
    assert payload["result"]["passed_tasks"] == 1
    assert payload["diagnosis"]["native_mobile_promotion"] is False
