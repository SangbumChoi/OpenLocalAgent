"""Integrity checks for the AgentNet-to-productivity state-machine bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m222-agentnet-continuation-stateful-productivity-bridge-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m222_receipt_is_self_hashed_and_fixture_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["runtime"]["environment_executed"] is True
    assert payload["runtime"]["public_benchmark"] is False
    assert payload["runtime"]["external_accounts_used"] is False
    assert payload["runtime"]["tool_pool_size"] == 63
    assert payload["oracle"]["task_complete_rate"] == 1.0
    assert payload["oracle"]["accepted_steps"] == 16


def test_m222_warm_partial_gain_does_not_promote_email_or_notion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = payload["warm"]
    random = payload["random_control"]
    assert warm["accepted_steps"] > random["accepted_steps"]
    assert warm["task_complete_rate"] == random["task_complete_rate"] == 0.2
    assert warm["by_family"]["email"]["task_complete_rate"] == 0.0
    assert warm["by_family"]["notion"]["task_complete_rate"] == 0.0
    assert warm["receipt"]["receipt_self_sha256"]
    assert random["receipt"]["receipt_self_sha256"]
    assert "do_not_promote" in payload["comparison"]["decision"]
