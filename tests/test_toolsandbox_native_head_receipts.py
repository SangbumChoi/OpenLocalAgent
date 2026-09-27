"""Regression checks for the current native ToolSandbox head ablation."""

import json
from pathlib import Path


def _load(name: str) -> dict[str, object]:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    return json.loads((root / name).read_text())


def test_current_head_children_have_matched_native_toolsandbox_receipts() -> None:
    warm = _load("m106-toolsandbox-native-warm-v1.json")
    random = _load("m107-toolsandbox-native-random-v1.json")
    comparison = _load("m108-toolsandbox-native-head-ablation-v1.json")
    assert warm["benchmark_id"] == random["benchmark_id"] == "toolsandbox"
    assert warm["environment_executed"] is True
    assert random["environment_executed"] is True
    assert warm["official_split_verified"] is False
    assert random["official_split_verified"] is False
    assert warm["task_count"] == random["task_count"] == 5
    assert warm["success_count"] == random["success_count"] == 1
    assert warm["success_rate"] == random["success_rate"] == 0.2
    assert comparison["kind"] == "localagent_toolsandbox_native_head_ablation_report"
    assert comparison["decision"] == "no_warm_native_advantage_on_bounded_toolsandbox_probe"
    assert comparison["warm_minus_random_success_rate_pp"] == 0.0
    assert "official split" in comparison["claim_boundary"]
