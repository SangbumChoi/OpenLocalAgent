import hashlib
import json
from pathlib import Path


def test_m547_mobilegym_receipt_is_checkpoint_bound_and_official_split() -> None:
    path = Path("docs/paper/results/raw/m547-m546-mobilegym-native-full-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["environment"]["official_split_verified"] is True
    assert payload["environment"]["task_count"] == 256
    assert payload["environment"]["errors"] == []
    assert payload["result"]["success_rate"] == 1 / 256
    assert payload["candidate_checkpoint"]["parameters"] < 100_000_000
    assert payload["decision"]["adoption"] == "retain_as_native_negative_control"
