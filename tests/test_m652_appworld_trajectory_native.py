import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m652-appworld-trajectory-native-v1.json")


def test_m652_native_trajectory_probe_is_fail_closed() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["native_replay"]["warm"]["summary"]["native_successes"] == 0
    assert payload["native_replay"]["random"]["summary"]["native_successes"] == 0
    assert payload["decision"]["promote_to_native_appworld_success"] is False
