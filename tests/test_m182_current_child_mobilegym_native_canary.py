import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m182-current-child-mobilegym-native-canary-v1.json")


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m182_mobilegym_canary_is_native_but_not_full_score() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    assert receipt["environment"]["executed"] is True
    assert receipt["environment"]["official_split_verified"] is True
    assert receipt["environment"]["test_task_count"] == 256
    assert receipt["canary"]["task_count"] == 1
    assert receipt["canary"]["runtime_errors"] == 0
    assert receipt["canary"]["success_rate"] == 0.0
    assert receipt["decision"] == "diagnostic_only"
