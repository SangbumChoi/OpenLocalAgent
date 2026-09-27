import hashlib
import json
from pathlib import Path


def test_m614_current_base_native_receipt_is_self_consistent_and_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m614-m585-toolsandbox-base-native-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["checkpoint"]["sha256"] == "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
    assert payload["task_count"] == 129
    assert payload["success_count"] == 30
    assert payload["official_split_verified"] is False
    assert payload["user_simulator_executed"] is False
    assert payload["external_api_called"] is False
    assert payload["selection"]["official_scenario_universe_count"] == 1032
    assert payload["analysis"]["by_category"]["INSUFFICIENT_INFORMATION"]["exact_rate"] == 13 / 14
    assert "not an official ToolSandbox leaderboard score" in payload["claim_boundary"]
