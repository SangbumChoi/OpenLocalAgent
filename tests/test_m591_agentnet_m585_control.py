"""Integrity checks for the m591 matched AgentNet projection control."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m591-agentnet-m585-warm-parent-control-v1.json"


def test_m591_receipt_binds_public_split_and_identical_projection() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["dataset"] == "xlangai/AgentNet"
    assert data["source_revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert data["source"]["normalized_eval_parent_records"] == 16
    assert data["source"]["normalized_eval_rows"] == 257
    assert data["evaluation"]["completeness_verified"] is True
    assert data["parent"]["prediction_sha256"] == data["warm_child"]["prediction_sha256"]
    assert data["parent"]["ground_truth_sha256"] == data["warm_child"]["ground_truth_sha256"]
    assert data["evaluation"]["warm_minus_parent"]["prediction_bytes_identical"] is True
    assert data["evaluation"]["warm_child"] == data["evaluation"]["parent"]
    assert "not AgentNetBench" in data["claim_boundary"]


def test_m591_receipt_self_hash() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
