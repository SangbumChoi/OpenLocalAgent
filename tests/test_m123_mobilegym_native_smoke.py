import json
from pathlib import Path


def test_m123_mobilegym_native_smoke_binds_registry_and_honest_boundary() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m123-mobilegym-native-runtime-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["registry"]["train_tasks"] == 160
    assert receipt["registry"]["test_tasks"] == 256
    assert receipt["registry"]["train_missing"] == []
    assert receipt["registry"]["test_missing"] == []
    assert receipt["registry"]["train_test_overlap"] == []
    assert receipt["runtime"]["http_status"] == 200
    assert receipt["runtime"]["sim_bridge"] is True
    assert receipt["runtime"]["state_initial"]["top_level_keys"] == ["apps", "os"]
    assert receipt["runtime"]["reset_hash_equal"] is False
    assert any("timestamp" in item["path"] for item in receipt["runtime"]["reset_diff_paths"])
    assert receipt["localagent_adaptation"]["model_invocations"] == 0
    assert receipt["localagent_adaptation"]["official_score"] is None
