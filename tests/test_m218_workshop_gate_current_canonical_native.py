from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "docs/paper/results/raw/m218-workshop-gate-current-canonical-native-v1.json"


def test_m218_joins_existing_native_receipts_without_overclaiming() -> None:
    payload = json.loads(GATE.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert len(payload["blocking_requirements"]) == 9
    checks = {check["requirement"]: check for check in payload["checks"]}
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
    assert checks["native:toolsandbox"]["status"] == "blocked"
    assert checks["native:toolsandbox"]["blockers"] == ["official_split_not_verified"]
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["artifacts:public_model_demo_manifest"]["status"] == "pass"
    assert payload["claim_boundary"].startswith("ready is false unless native benchmark receipts")
