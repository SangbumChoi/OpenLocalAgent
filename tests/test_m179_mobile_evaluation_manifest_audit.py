import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m179-mobile-evaluation-manifest-audit-v1.json")


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m179_manifest_audit_is_hash_pinned_and_eval_only() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    sources = {row["id"]: row for row in receipt["sources"]}

    ios = sources["iosworld"]
    assert ios["evaluation_manifests"] == [
        {
            "path": "tasks.json",
            "url": "https://raw.githubusercontent.com/ljang0/iOSWorld/e91f4cb2ef4c9dd48fef83a894477b41fd5e209d/tasks.json",
            "bytes": 192020,
            "sha256": "42c0b5558fde0e71193b79fa3222e59d6f8f9be72e8151e260ddb0d6f99a76bb",
            "rows": 133,
            "category_counts": {"single_app": 27, "multi_app": 60, "memory": 46},
            "app_count": 26,
            "task_text_retained": False,
        }
    ]
    assert ios["acquisition"]["training_rows_admitted"] == 0
    assert ios["acquisition"]["native_runner_executed"] is False

    mobile = sources["mobile_safety_bench"]
    assert mobile["evaluation_manifests"][0]["rows"] == 90
    assert mobile["evaluation_manifests"][1]["rows"] == 3
    assert mobile["manifest_reconciliation"] == {
        "paper_suite_tasks": 100,
        "source_task_table_rows": 90,
        "qa_analysis_rows": 3,
        "interpretation": "The public task table and QA analysis file are recorded as separate artifacts; their row counts must not be summed or substituted for the paper's 100-task suite without the upstream runner's task assembly rules.",
    }
    assert mobile["acquisition"]["repository_payload_retained"] is False
    assert mobile["acquisition"]["training_rows_admitted"] == 0

    assert receipt["localagent_adaptation"]["training_rows_added"] == 0
    assert receipt["localagent_adaptation"]["native_runs"] == 0
    assert receipt["localagent_adaptation"]["official_scores_added"] == 0
