import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m649-appworld-runtime-pool-v1.json")


def test_m649_full_pool_replays_actions_but_keeps_native_boundary() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["diagnostic"]["matched_retrieve_k"] == 100
    assert payload["native_replay"]["summary"]["action_replayed"] == 6
    assert payload["native_replay"]["summary"]["native_successes"] == 0
    assert payload["decision"]["promote_to_appworld_success"] is False
