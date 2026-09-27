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


def test_m146_mobilegym_current_checkpoint_binds_official_text_projection() -> None:
    path = Path("docs/paper/results/raw/m146-mobilegym-native-current-checkpoint-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source"]["repository"] == "https://github.com/Purewhiter/mobilegym"
    assert receipt["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert receipt["environment_executed"] is True
    assert receipt["official_split_verified"] is True
    assert receipt["native_receipt_eligible"] is True
    assert receipt["task_count"] == 256
    assert receipt["official_test_task_count"] == 256
    assert receipt["passed_tasks"] == 13
    assert receipt["failed_tasks"] == 243
    assert receipt["success_rate"] == 13 / 256
    assert receipt["errors"] == []
    assert receipt["run"]["max_steps"] == 2
    assert receipt["run"]["full_official_test_split"] is True
    assert receipt["observation_mode"] == "text_projection"
    assert receipt["vision_used"] is False
    assert receipt["tool_counts"] == {"mobile_submit_answer": 215}
    assert sum(1 for task in receipt["task_results"] if not task["tool_names"]) == 41
    assert receipt["checkpoint_sha256"] == (
        "8cc3ee42ed38b830b9b04935e156f80d166abeece2dd0c37184ee4d692de7eb1"
    )
