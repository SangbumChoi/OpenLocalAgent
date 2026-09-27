"""Integrity checks for the m588 warm-child MobileGym full split receipt."""

import hashlib
import json
from pathlib import Path

from localagent.eval.workshop_gate import build_workshop_gate


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m588-m585-mobilegym-native-full-v1.json"


def test_m588_receipt_binds_warm_child_and_official_split() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["benchmark_id"] == "mobilegym"
    assert data["environment_executed"] is True
    assert data["official_split_verified"] is True
    assert data["native_receipt_eligible"] is True
    assert data["task_count"] == 256
    assert data["success_rate"] == 1 / 256
    assert data["checkpoint_sha256"] == (
        "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
    )
    assert data["result"]["passed_tasks"] == 1
    assert data["result"]["failed_tasks"] == 255
    assert data["comparison"]["warm_minus_parent_success_pp"] == 0.0
    assert data["vision_used"] is False


def test_m588_receipt_self_hash_and_gate_consumption() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
    report = build_workshop_gate(
        "configs/data/realistic-agent-eval.catalog.yaml",
        repo_root=ROOT,
        native_receipts={"mobilegym": str(RECEIPT)},
    )
    checks = {item["requirement"]: item for item in report["checks"]}
    assert checks["native:mobilegym"]["status"] == "pass"
    assert report["ready"] is False
