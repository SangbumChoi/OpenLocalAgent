import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m657-appworld-rich-state-v1.json")


def test_m657_rich_state_is_fail_closed_and_records_truncation() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["rich_observations"] is True
    assert payload["benchmark"]["eval_truncated_rows"] == 18
    assert payload["native_replay"]["warm"]["summary"]["native_successes"] == 0
    assert payload["decision"]["promote_to_native_success"] is False
