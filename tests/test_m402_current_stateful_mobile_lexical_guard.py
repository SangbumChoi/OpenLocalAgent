"""Integrity checks for the current stateful mobile/action guard receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m402-current-stateful-mobile-lexical-guard-v1.json"
)


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m402_is_current_child_bound_and_self_hashed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["checkpoint"]["sha256"] == (
        "c53c79ad068878966ab897b5bf708d34d832e960f64a303ec059c2e0c8b90bfd"
    )
    assert payload["oracle"]["task_complete_rate"] == 1.0
    assert payload["model"]["task_complete_rate"] == 1.0
    assert payload["model"]["accepted_steps"] == 16
    assert payload["model"]["attempts"] == 17
    assert payload["configuration"]["tool_pool_size"] == 63
    assert payload["runtime"]["public_benchmark"] is False
    assert payload["runtime"]["external_accounts_used"] is False
    assert "AndroidWorld" in payload["claim_boundary"]
    assert "benchmark result" in payload["claim_boundary"]
