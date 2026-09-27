"""Integrity checks for the current AgentNet-to-MCPMark routing bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m223-agentnet-continuation-mcpmark-routing-bridge-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m223_receipt_is_self_hashed_and_public_proxy_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["dataset"]["url"] == "https://github.com/eval-sys/mcpmark"
    assert payload["dataset"]["standard"]["rows"] == 169
    assert payload["dataset"]["easy"]["rows"] == 70
    assert payload["dataset"]["task_text_retained"] is False
    assert payload["dataset"]["mcp_servers_executed"] is False
    assert payload["dataset"]["verifiers_executed"] is False
    assert "official leaderboard" in payload["claim_boundary"]


def test_m223_warm_route_is_not_promoted_to_native_mcp_control() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    reports = payload["reports"]
    assert reports["warm_standard"]["route_accuracy"] == 0.1952662721893491
    assert reports["random_standard"]["route_accuracy"] == 0.22485207100591717
    assert reports["warm_easy"]["route_accuracy"] == 0.22857142857142856
    assert reports["random_easy"]["route_accuracy"] == 0.18571428571428572
    assert reports["warm_standard"]["by_service"]["notion"] > 0.0
    assert reports["warm_standard"]["by_service"]["filesystem"] == 0.0
    assert "do_not_promote" in payload["comparison"]["decision"]
