import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m636-m626-toolsandbox-matched-random-control-v1.json")
RANDOM_SHA = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"


def test_m636_random_control_is_current_and_bounded() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark_id"] == "toolsandbox"
    assert payload["checkpoint"]["sha256"] == RANDOM_SHA
    assert payload["random_control"]["task_count"] == 25
    assert payload["random_control"]["success_count"] == 4
    assert payload["matched_warm"]["success_count"] == 5
    assert payload["protocol"]["official_split_verified"] is False
