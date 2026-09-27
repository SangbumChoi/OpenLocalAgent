from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m369-workshop-gate-current-toolsandbox-interactive-v1.json"


def test_m369_gate_is_self_hashed_and_fail_closed() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == expected
    assert receipt["kind"] == "localagent_workshop_publication_gate"
    assert receipt["ready"] is False
    assert receipt["current_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    blocked = {item["requirement"]: item["blockers"] for item in receipt["blocking_requirements"]}
    assert blocked["native:toolsandbox"] == ["official_split_not_verified"]
    assert blocked["artifacts:public_model_demo_manifest"] == ["current_checkpoint_not_bound"]
    assert "native:androidworld" in blocked
