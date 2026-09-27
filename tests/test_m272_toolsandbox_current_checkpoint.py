"""Integrity checks for the current ToolSandbox projection diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m272-toolsandbox-current-checkpoint-text-projection-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_m272_receipt_is_self_hashed_and_explicitly_offline() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["dataset"]["source_url"] == "https://github.com/apple/ToolSandbox"
    assert payload["dataset"]["official_split_verified"] is False
    assert payload["dataset"]["simulator_executed"] is False
    assert payload["dataset"]["verifiers_executed"] is False
    assert payload["metrics"]["tool_exact_rate"] == 0.55
    assert payload["metrics"]["arguments_exact_rate"] == 0.30
    assert payload["metrics"]["schema_valid_rate"] == 0.95
    assert "not an official ToolSandbox score" in payload["claim_boundary"]
