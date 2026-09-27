import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m163-mobile-grounding-source-audit-v1.json")


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m163_source_audit_is_hash_pinned_and_non_native() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    sources = {row["id"]: row for row in receipt["sources"]}

    assert sources["knowu_bench"]["repository_revision"]["commit"] == (
        "c03a825991ede13add6631f2ed19b90755930dc6"
    )
    assert sources["knowu_bench"]["benchmark_contract"]["registered_tasks"] == 192
    assert sources["knowu_bench"]["benchmark_contract"]["app_coverage"] == 23
    assert sources["knowu_bench"]["benchmark_contract"]["hidden_profiles"] is True

    assert sources["appagent_benchmark"]["benchmark_contract"] == {
        "tasks": 45,
        "android_apps": 9,
        "action_family": ["tap", "swipe", "text_input", "back", "home"],
        "test_manifest_text_retained": False,
        "test_manifest_sha256_only": True,
    }
    assert sources["groundcua"]["benchmark_contract"]["screenshots"] == 56000
    assert sources["groundcua"]["benchmark_contract"]["screenshot_payload_retained"] is False
    assert sources["ui_tars_action_contract"]["benchmark_contract"]["mobile_actions"] == [
        "long_press",
        "open_app",
        "press_home",
        "press_back",
    ]

    assert receipt["localagent_adaptation"] == {
        "training_rows_added": 0,
        "native_runs": 0,
        "webgpu_runs": 0,
        "official_scores_added": 0,
        "reason": "This pass records public provenance and modality boundaries; the current text-first WebGPU model has no native Android/desktop vision runtime and no authenticated benchmark services.",
    }
