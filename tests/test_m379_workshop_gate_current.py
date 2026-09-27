from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m379-workshop-gate-current-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m379_gate_binds_current_parent_and_new_unified_weight_report() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["kind"] == "localagent_workshop_publication_gate"
    assert receipt["ready"] is False
    assert receipt["current_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    checks = {item["requirement"]: item for item in receipt["checks"]}
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["training:rl_preflight"]["status"] == "pass"
    assert checks["native:toolsandbox"]["blockers"] == ["official_split_not_verified"]
    assert checks["artifacts:public_model_demo_manifest"]["blockers"] == [
        "current_checkpoint_not_bound"
    ]
    blocked = {item["requirement"] for item in receipt["blocking_requirements"]}
    assert "native:androidworld" in blocked
    assert "native:mcpmark" in blocked
