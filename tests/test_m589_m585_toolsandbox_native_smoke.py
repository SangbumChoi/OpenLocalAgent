"""Integrity checks for the m589 current warm-child ToolSandbox smoke."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m589-m585-toolsandbox-native-smoke-v1.json"


def test_m589_receipt_is_current_and_explicitly_non_official() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["benchmark_id"] == "toolsandbox"
    assert data["environment_executed"] is True
    assert data["official_split_verified"] is False
    assert data["user_simulator_executed"] is False
    assert data["verifier_executed"] is True
    assert data["external_api_called"] is False
    assert data["task_count"] == data["success_count"] == 3
    assert data["success_rate"] == 1.0
    assert data["checkpoint_sha256"] == (
        "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
    )
    assert all(item["similarity"] == 1.0 for item in data["scenarios"].values())
    assert "not an official ToolSandbox leaderboard score" in data["claim_boundary"]


def test_m589_receipt_self_hash() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
