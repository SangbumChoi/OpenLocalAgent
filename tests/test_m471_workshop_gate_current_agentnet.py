"""The m471 gate joins current AgentNet transfer evidence without weakening native blockers."""

import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m471-workshop-gate-current-agentnet-v1.json")
CURRENT_CHECKPOINT_SHA256 = (
    "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
)


def test_m471_gate_is_current_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_workshop_publication_gate"
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == CURRENT_CHECKPOINT_SHA256
    checks = {row["requirement"]: row for row in payload["checks"]}
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
    assert checks["native:toolsandbox"]["blockers"] == ["official_split_not_verified"]
    assert checks["native:mcpmark"]["blockers"] == ["official_split_not_verified"]
    assert checks["artifacts:public_model_demo_manifest"]["blockers"] == ["manifest_not_supplied"]
    assert len(payload["blocking_requirements"]) == 10
