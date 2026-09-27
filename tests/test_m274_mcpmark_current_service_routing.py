"""Integrity checks for the current MCPMark routing proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m274-mcpmark-current-service-routing-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_m274_receipt_is_self_hashed_and_keeps_native_boundary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["dataset"]["url"] == "https://github.com/eval-sys/mcpmark"
    assert payload["dataset"]["mcp_servers_executed"] is False
    assert payload["dataset"]["verifiers_executed"] is False
    standard = payload["standard_suite"]["current_parent"]
    assert standard["rows"] == 169
    assert standard["route_correct"] == 25
    assert standard["by_service"]["notion"]["route_correct"] == 0
    assert standard["by_service"]["playwright"]["route_correct"] == 25
    assert payload["easy_suite_current_parent"]["notion_route_correct"] == 0
    assert "not an official MCPMark score" in payload["claim_boundary"]
