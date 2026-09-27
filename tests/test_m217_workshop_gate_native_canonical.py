from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "docs/paper/results/raw/m217-workshop-gate-native-canonical-v1.json"


def test_m217_canonical_webgpu_gate_is_bound_but_not_ready() -> None:
    payload = json.loads(GATE.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert len(payload["blocking_requirements"]) == 12
    checks = {check["requirement"]: check for check in payload["checks"]}
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["artifacts:public_model_demo_manifest"]["status"] == "blocked"
    assert payload["claim_boundary"].startswith("ready is false unless native benchmark receipts")
