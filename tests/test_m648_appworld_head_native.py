import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m648-appworld-head-native-v1.json")


def test_m648_head_replay_keeps_teacher_forcing_and_native_boundaries() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["head_eval_rows"] == 18
    assert payload["benchmark"]["native_tasks"] == 6
    assert payload["head_adaptation"]["matched"]["warm_route_after"] == 1.0
    assert payload["head_adaptation"]["matched"]["warm_selector_after"] == 1.0
    assert payload["native_replay"]["baseline"]["summary"]["native_successes"] == 0
    assert payload["native_replay"]["head_adapted"]["summary"]["native_successes"] == 0
    assert payload["native_replay"]["head_adapted"]["summary"]["native_api_calls"] == 0
    assert payload["decision"]["promote_head_to_native_success"] is False
