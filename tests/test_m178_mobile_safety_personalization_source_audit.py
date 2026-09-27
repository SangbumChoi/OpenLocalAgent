import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m178-mobile-safety-personalization-source-audit-v1.json"
)


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m178_source_audit_is_hash_pinned_and_non_native() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    sources = {row["id"]: row for row in receipt["sources"]}

    mobile = sources["mobile_safety_bench"]
    assert mobile["repository_revision"]["commit"] == (
        "bc5e0579626a280c4f551261abcb721442ff92ea"
    )
    assert mobile["benchmark_contract"]["tasks"] == 100
    assert mobile["benchmark_contract"]["safety_tasks"] == 42
    assert mobile["benchmark_contract"]["indirect_prompt_injection_tasks"] == 8
    assert mobile["runtime"]["android_emulator"] is True
    assert mobile["runtime"]["local_task_runner_executed"] is False

    ios = sources["iosworld"]
    assert ios["repository_revision"]["commit"] == (
        "e91f4cb2ef4c9dd48fef83a894477b41fd5e209d"
    )
    assert ios["benchmark_contract"] == {
        "apps": 26,
        "tasks": 133,
        "single_app_tasks": 27,
        "multi_app_tasks": 60,
        "memory_personalization_tasks": 46,
        "persistent_seeded_identity": True,
        "cross_app_state": True,
        "observations": ["screenshot", "optional_accessibility_xml"],
        "evaluation": ["task_rubrics", "trajectory_artifacts", "final_state"],
        "optional_mcp_mode": True,
    }
    assert ios["runtime"]["ios_simulator"] is True
    assert ios["runtime"]["local_task_runner_executed"] is False

    assert receipt["localagent_adaptation"] == {
        "training_rows_added": 0,
        "native_runs": 0,
        "webgpu_runs": 0,
        "official_scores_added": 0,
        "reason": "The current text-first WebGPU model cannot claim screenshot-grounded mobile control, persistent iOS identity, or safety behavior without the release-matched Android/iOS runtimes and evaluators.",
    }
