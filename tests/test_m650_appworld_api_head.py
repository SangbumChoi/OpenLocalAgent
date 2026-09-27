import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m650-appworld-api-head-v1.json")


def test_m650_api_head_is_diagnostic_not_native_success() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["api_head"]["metrics"]["eval"]["accuracy"] == 0.6
    assert payload["native_replay"]["summary"]["native_successes"] == 0
    assert payload["decision"]["promote_to_native_success"] is False
