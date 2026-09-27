import json
from pathlib import Path


def test_m121_source_audit_is_hash_pinned_and_non_native() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m121-mobilegym-osworld-source-audit-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["sources"][0]["repository_revision"]["commit"] == (
        "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    )
    assert receipt["sources"][0]["benchmark_contract"]["simulated_apps"] == 28
    assert receipt["sources"][0]["benchmark_contract"]["task_templates_total"] == 416
    assert receipt["sources"][0]["benchmark_contract"]["test_tasks"] == 256
    assert receipt["sources"][0]["license"]["benchmark_data"] == "CC-BY-NC-4.0"
    assert receipt["sources"][1]["release"]["task_count"] == 108
    assert receipt["sources"][1]["release"]["name"] == "osworld-v2-2026.06.24"
    assert receipt["localagent_adaptation"]["training_rows_added"] == 0
    assert receipt["localagent_adaptation"]["native_runs"] == 0
    assert receipt["localagent_adaptation"]["official_scores_added"] == 0
