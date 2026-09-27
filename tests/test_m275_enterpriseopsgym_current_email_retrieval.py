"""Integrity checks for the current EnterpriseOps-Gym email retrieval receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m275-enterpriseopsgym-current-email-retrieval-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_m275_receipt_is_self_hashed_and_records_negative_transfer() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert (
        payload["dataset"]["url"]
        == "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"
    )
    assert payload["dataset"]["records"] == 67
    assert payload["results"]["current_parent"]["hit_at_1"] == 0.208955223880597
    assert payload["results"]["m273_warm_child"]["hit_at_1"] == 0.2537313432835821
    assert payload["results"]["m273_random_child"]["hit_at_1"] == 0.3582089552238806
    assert "negative-transfer" in payload["interpretation"]
    assert "No MCP server" in payload["claim_boundary"]
