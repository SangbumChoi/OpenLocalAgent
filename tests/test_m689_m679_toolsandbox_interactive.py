import hashlib
import json
from pathlib import Path


def test_m689_interactive_toolsandbox_receipt_is_current_and_negative() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m689-m679-toolsandbox-interactive-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["protocol"] == "bounded_multi_step_scripted_user"
    assert payload["success_count"] == 27
    assert payload["category_metrics"]["MULTIPLE_TOOL_CALL"]["exact_count"] == 0
    assert payload["category_metrics"]["STATE_DEPENDENCY"]["exact_count"] == 0
