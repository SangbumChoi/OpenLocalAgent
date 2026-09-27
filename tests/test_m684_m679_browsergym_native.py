import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m684_browsergym_binds_current_m679_and_rejects_promotion() -> None:
    path = ROOT / "docs/paper/results/raw/m684-m679-browsergym-native-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["official_split_verified"] is True
    assert payload["task_count"] == 240
    assert payload["result"]["passed_tasks"] == 5
    assert payload["result"]["action_errors"] == 0
    assert payload["decision"]["native_browser_promotion"] is False
    assert payload["result"]["vision_used"] is False
