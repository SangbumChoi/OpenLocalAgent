import hashlib
import json
from pathlib import Path


def test_m538_candidate_binds_training_webgpu_and_toolsandbox() -> None:
    path = Path("docs/paper/results/raw/m538-warm-realistic-candidate-webgpu-toolsandbox-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["candidate"]["checkpoint"]["sha256"] == (
        "3b6325676bb1812ea34ddf55326a1a9cdaa62970d38309350451eea520930a1b"
    )
    assert payload["candidate"]["training_parent_sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["webgpu"]["backend"] == "webgpu"
    assert payload["webgpu"]["environment_executed"] is True
    assert payload["webgpu"]["exact_actions"] == payload["webgpu"]["evaluated_cases"] == 3
    assert payload["toolsandbox"]["success_count"] == 3
    assert payload["toolsandbox"]["official_split_verified"] is False
    assert payload["trajectory"]["bundle_checkpoint_sha256"] == payload["candidate"]["checkpoint"]["sha256"]
    assert payload["trajectory"]["pass_at_1"] == 1.0
    assert payload["trajectory"]["steps"] == 13


def test_m538_toolsandbox_receipt_is_self_hashed() -> None:
    path = Path("docs/paper/results/raw/m538-toolsandbox-warm-native-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["checkpoint"]["sha256"] == (
        "3b6325676bb1812ea34ddf55326a1a9cdaa62970d38309350451eea520930a1b"
    )
    assert payload["success_count"] == payload["task_count"] == 3
