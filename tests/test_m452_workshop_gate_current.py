"""The current publication gate must remain fail-closed until external blockers are resolved."""

import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m452-workshop-gate-current-v1.json")
CURRENT_CHECKPOINT_SHA256 = (
    "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
)


def test_m452_current_gate_is_bound_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_workshop_publication_gate"
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == CURRENT_CHECKPOINT_SHA256
    checks = {row["requirement"]: row for row in payload["checks"]}
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["native:mcpmark"]["status"] == "blocked"
    assert "official_split_not_verified" in checks["native:mcpmark"]["blockers"]
    assert checks["artifacts:public_model_demo_manifest"]["status"] == "blocked"
    assert checks["training:rl_preflight"]["status"] == "blocked"
    assert len(payload["blocking_requirements"]) >= 10
