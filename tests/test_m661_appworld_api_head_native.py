import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m661-appworld-api-head-native-v1.json")


def test_m661_api_head_native_replay_remains_fail_closed() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert recorded == expected
    assert payload["api_head"]["metrics"]["eval"]["accuracy"] == 0.4285714328289032
    assert payload["native_replay"]["api_head"]["summary"]["native_successes"] == 0
    assert payload["decision"]["promote_to_native_success"] is False
