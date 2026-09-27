from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m253-workshop-gate-tau2-catalog-refresh-v1.json"


def test_m253_gate_is_current_and_remains_fail_closed() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["ready"] is False
    assert len(receipt["blocking_requirements"]) == 9
    assert receipt["catalog"]["entries"] == 40
    blockers = {item["requirement"] for item in receipt["blocking_requirements"]}
    assert "native:toolsandbox" in blockers
    assert "native:tau_bench" not in blockers
    artifact_check = next(
        item for item in receipt["checks"] if item["requirement"] == "artifacts:public_model_demo_manifest"
    )
    assert artifact_check["status"] == "pass"
