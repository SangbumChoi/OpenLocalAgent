import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m262-mobilegym-native-current-browser-context-v1.json")


def test_m262_current_checkpoint_mobilegym_is_official_split_and_hash_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["source"]["repository"] == "https://github.com/Purewhiter/mobilegym"
    assert payload["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is True
    assert payload["native_receipt_eligible"] is True
    assert payload["task_count"] == payload["official_test_task_count"] == 256
    assert payload["passed_tasks"] == 1
    assert payload["failed_tasks"] == 255
    assert payload["success_rate"] == 1 / 256
    assert payload["errors"] == []
    assert payload["run"]["full_official_test_split"] is True
    assert payload["run"]["max_steps"] == 2
    assert payload["observation_mode"] == "text_projection"
    assert payload["vision_used"] is False
    assert payload["checkpoint_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["suite_summary"]["crossapp_life"] == {
        "tasks": 29,
        "passed": 1,
        "success_rate": 1 / 29,
    }
    assert payload["tool_counts"] == {
        "mobile_input_text": 146,
        "mobile_press_enter": 110,
    }
