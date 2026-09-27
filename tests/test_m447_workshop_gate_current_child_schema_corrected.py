"""Integrity checks for the current gate after the MCP schema correction."""

import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m447-workshop-gate-current-child-mcpmark-schema-corrected-v1.json"
)


def test_m447_gate_uses_schema_corrected_mcpmark_and_remains_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    mcp = next(item for item in payload["checks"] if item["requirement"] == "native:mcpmark")
    assert mcp["status"] == "blocked"
    assert mcp["blockers"] == ["official_split_not_verified"]
    assert any(item["requirement"] == "training:rl_preflight" for item in payload["blocking_requirements"])
    assert any(
        item["requirement"] == "artifacts:public_model_demo_manifest"
        for item in payload["blocking_requirements"]
    )
