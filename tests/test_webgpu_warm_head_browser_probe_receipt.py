"""Regression checks for the current warm-head browser probe, including its negative result."""

import json
from pathlib import Path


def test_warm_head_browser_probe_records_realistic_dispatch_regression() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipt = json.loads((root / "m110-webgpu-warm-head-browser-probe-v1.json").read_text())
    assert receipt["kind"] == "localagent_webgpu_warm_head_browser_probe"
    assert receipt["verified"] is True
    assert receipt["bundle"]["source_receipt"] == "m109-webgpu-warm-head-deploy-v1.json"
    assert receipt["deployment"]["native_adapter_verified"] is False
    assert receipt["outcome"] == {
        "case_count": 4,
        "exact_dispatch_count": 0,
        "exact_dispatch_rate": 0.0,
        "status": "blocked_current_bundle_dispatch_regression",
    }
    assert all(case["observed_tool"] == "click" for case in receipt["cases"])
    assert all(case["exact_match"] is False for case in receipt["cases"])
