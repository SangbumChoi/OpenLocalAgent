import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m635-m626-toolsandbox-native-base-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m635_toolsandbox_is_current_and_explicitly_nonofficial() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark_id"] == "toolsandbox"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 129
    assert payload["success_count"] == 27
    assert payload["checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["category_summary"]["INSUFFICIENT_INFORMATION"]["exact"] == 26
    assert payload["category_summary"]["CANONICALIZATION"]["exact"] == 0
