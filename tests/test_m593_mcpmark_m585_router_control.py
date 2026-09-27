"""Integrity checks for the m593 matched MCPMark service-router control."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m593-mcpmark-m585-warm-parent-router-control-v1.json"


def test_m593_receipt_binds_mcpmark_revision_and_notion_negative_control() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["dataset"] == "MCPMark"
    assert data["source_revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert data["source"]["standard"]["rows"] == 169
    assert data["source"]["easy"]["rows"] == 70
    assert data["source"]["mcp_servers_executed"] is False
    assert data["parent"]["standard"] == data["warm_child"]["standard"]
    assert data["parent"]["easy"] == data["warm_child"]["easy"]
    assert data["parent"]["by_service_standard"] == data["warm_child"]["by_service_standard"]
    assert data["warm_child"]["by_service_standard"]["notion"] == [0, 28]
    assert data["transfer_delta"]["standard_route_accuracy_pp"] == 0.0
    assert "not an official MCPMark result" in data["claim_boundary"]


def test_m593_receipt_self_hash() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
