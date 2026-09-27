import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m688_toolsandbox_current_receipt_is_complete_and_fail_closed() -> None:
    path = ROOT / "docs/paper/results/raw/m688-m679-toolsandbox-native-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["environment_executed"] is True
    assert payload["verifier_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 129
    assert payload["success_count"] == 30
    assert payload["category_metrics"]["INSUFFICIENT_INFORMATION"]["exact_count"] == 26
    assert payload["category_metrics"]["STATE_DEPENDENCY"]["exact_count"] == 0
