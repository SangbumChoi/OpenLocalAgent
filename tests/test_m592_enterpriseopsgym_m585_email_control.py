"""Integrity checks for the m592 current warm-child email retrieval control."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m592-enterpriseopsgym-m585-warm-parent-email-control-v1.json"


def test_m592_receipt_binds_public_email_parquets_and_transfer_delta() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["dataset"] == "ServiceNow-AI/EnterpriseOps-Gym"
    assert data["dataset_revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert data["source"]["rows"] == 67
    assert data["source"]["verifiers_dropped"] is True
    assert data["source"]["server_configuration_dropped"] is True
    assert data["parent"]["hit_at_1"] == 0.43283582089552236
    assert data["warm_child"]["hit_at_1"] == 0.5671641791044776
    assert data["transfer_delta"]["hit_at_1_pp"] == 13.432835820895525
    assert data["transfer_delta"]["hit_at_3_pp"] == 0.0
    assert data["transfer_delta"]["hit_at_5_pp"] == 0.0
    assert "official leaderboard score" in data["claim_boundary"]


def test_m592_receipt_self_hash() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
