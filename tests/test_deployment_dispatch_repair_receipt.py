"""Regression checks for the m111 deployment dispatch repair and browser evidence."""

import json
from pathlib import Path


def test_m111_repair_binds_public_rows_weights_and_browser_contract() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipt = json.loads((root / "m111-deployment-dispatch-repair-v1.json").read_text())
    assert receipt["kind"] == "localagent_deployment_dispatch_repair_browser_verification"
    assert receipt["verified"] is True
    assert receipt["checkpoint"]["parameters"] == 10524544
    assert receipt["export"]["parity_hard_gate"] is True
    assert len(receipt["training"]["public_sources"]) == 3
    assert receipt["training"]["public_train_rows"] == 4731
    assert receipt["training"]["public_eval_rows"] == 1044
    assert receipt["training"]["offline_mixed_probe"]["warm_canonical_tool_exact"] == 4
    assert receipt["training"]["offline_mixed_probe"]["random_canonical_tool_exact"] == 4
    cases = receipt["browser_probe"]["single_step_cases"]
    assert len(cases) == 5
    assert receipt["browser_probe"]["single_step_exact_tool_count"] == 5
    assert all(case["tool_exact"] for case in cases)
    planner = receipt["browser_probe"]["planner_case"]
    assert planner["observed_tools"] == ["web_search", "notion_write"]
    assert planner["sequence_exact"] is True
    assert receipt["browser_probe"]["native_adapter_verified"] is False
    assert receipt["browser_probe"]["external_action_executed"] is False
