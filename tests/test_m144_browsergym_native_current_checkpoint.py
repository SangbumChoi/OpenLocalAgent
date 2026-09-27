import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m144_browsergym_current_checkpoint_binds_official_negative_result() -> None:
    path = Path("docs/paper/results/raw/m144-browsergym-native-current-checkpoint-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source_url"] == "https://github.com/ServiceNow/BrowserGym"
    assert receipt["environment_executed"] is True
    assert receipt["official_split_verified"] is True
    assert receipt["task_count"] == 240
    assert receipt["success_count"] == 0
    assert receipt["success_rate"] == 0.0
    assert receipt["steps"] == 2400
    assert receipt["grounded_action_count"] == 0
    assert receipt["action_error_count"] == 0
    assert receipt["noop_step_count"] == 2400
    assert receipt["decision"]["native_browser_gate_eligible"] is True
    assert receipt["decision"]["checkpoint_promoted"] is False
    assert receipt["previous_comparison"]["success_count"] == 5
    assert receipt["previous_comparison"]["official_split_verified"] is True
