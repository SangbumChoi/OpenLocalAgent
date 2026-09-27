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


def test_m133_mobilegym_full_official_text_eval_is_hash_bound_and_redacted() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m133-mobilegym-native-text-eval-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["kind"] == "localagent_mobilegym_native_text_eval"
    assert receipt["benchmark_id"] == "mobilegym"
    assert receipt["environment_executed"] is True
    assert receipt["official_split"] == "test"
    assert receipt["official_split_verified"] is True
    assert receipt["native_receipt_eligible"] is True
    assert receipt["task_count"] == 256
    assert receipt["official_test_task_count"] == 256
    assert receipt["passed_tasks"] == 13
    assert receipt["success_rate"] == 13 / 256
    assert receipt["observation_mode"] == "text_projection"
    assert receipt["vision_used"] is False
    assert receipt["errors"] == []
    task_results = receipt["task_results"]
    assert len(task_results) == 256
    assert len({row["task_id"] for row in task_results}) == 256
    assert receipt["tool_counts"] == {
        "mobile_open_app": 1,
        "mobile_submit_answer": 200,
    }
    for row in task_results:
        assert "prompt" not in row
        assert "arguments" not in row
        assert "expected" not in row["judge"]
        assert "actual" not in row["judge"]
        assert len(row["trace_sha256"]) == 64
