import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m659-appworld-compact-state-v1.json")


def test_m659_compact_state_records_context_fit_and_native_boundary() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["eval_tokens_over_window"] == 3
    assert payload["native_replay"]["warm"]["summary"]["action_replayed"] == 5
    assert payload["decision"]["promote_to_native_success"] is False
