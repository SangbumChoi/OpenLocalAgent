"""Integrity checks for the gate refresh after the MCPMark Playwright run."""

import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m441-workshop-gate-current-child-mcpmark-playwright-v1.json"
)


def test_m441_gate_binds_current_child_and_keeps_publish_blocked() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    blocking = {item["requirement"]: item["blockers"] for item in payload["blocking_requirements"]}
    assert "official_split_not_verified" in blocking["native:mcpmark"]
    assert payload["checks"][-4]["requirement"] == "webgpu:native_capability_and_latency"
    assert payload["checks"][-4]["status"] == "pass"
    assert any(item["requirement"] == "training:rl_preflight" for item in payload["blocking_requirements"])
    assert any(item["requirement"] == "artifacts:public_model_demo_manifest" for item in payload["blocking_requirements"])

