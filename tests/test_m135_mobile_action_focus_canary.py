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


def test_m135_focus_canary_is_hash_bound_and_not_promoted() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m135-mobile-action-focus-canary-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["kind"] == "localagent_mobile_action_focus_canary"
    assert receipt["benchmark_id"] == "mobilegym"
    assert receipt["native_canary"]["environment_executed"] is True
    assert receipt["native_canary"]["official_split_verified"] is True
    assert receipt["native_canary"]["selected_task_count"] == 20
    assert receipt["native_canary"]["passed_tasks"] == 1
    assert receipt["native_canary"]["success_rate"] == 0.05
    assert receipt["native_canary"]["vision_used"] is False
    assert receipt["promotion_decision"]["promoted"] is False
    assert receipt["native_canary"]["tool_counts"] == {"mobile_submit_answer": 14}
