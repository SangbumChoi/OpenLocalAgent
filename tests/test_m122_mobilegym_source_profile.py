import json
from pathlib import Path


def test_m122_mobilegym_source_profile_binds_official_disjoint_splits() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m122-mobilegym-source-split-profile-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert receipt["source"]["archive"]["bytes"] == 185182209
    assert receipt["splits"]["train"]["tasks"] == 160
    assert receipt["splits"]["test"]["tasks"] == 256
    assert receipt["integrity"]["train_test_overlap"] == []
    assert receipt["integrity"]["train_test_unique_tasks"] == 416
    assert receipt["integrity"]["all_split_task_ids_unique"] is False
    assert receipt["localagent_adaptation"]["training_rows_added"] == 0
    assert receipt["localagent_adaptation"]["task_text_retained"] is False
