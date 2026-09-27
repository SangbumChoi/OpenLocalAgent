import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m189-current-m180-mobilegym-native-full-v1.json")


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    return actual


def test_m189_current_m180_mobilegym_is_a_complete_native_text_receipt() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    assert receipt["source"]["repository"] == "https://github.com/Purewhiter/mobilegym"
    assert receipt["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert receipt["environment_executed"] is True
    assert receipt["official_split"] == "test"
    assert receipt["official_split_verified"] is True
    assert receipt["native_receipt_eligible"] is True
    assert receipt["task_count"] == 256
    assert receipt["official_test_task_count"] == 256
    assert receipt["passed_tasks"] == 1
    assert receipt["failed_tasks"] == 255
    assert receipt["success_rate"] == 1 / 256
    assert receipt["errors"] == []
    assert receipt["run"]["full_official_test_split"] is True
    assert receipt["run"]["max_steps"] == 2
    assert receipt["observation_mode"] == "text_projection"
    assert receipt["vision_used"] is False
    assert len(receipt["task_results"]) == 256
    assert receipt["checkpoint_sha256"] == (
        "10827649f07182a08f8c11104d4713e76b8acaddc73020c5a4c77950de7b23a0"
    )
