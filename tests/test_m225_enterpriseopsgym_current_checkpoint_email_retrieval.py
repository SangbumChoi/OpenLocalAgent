"""Integrity checks for the current EnterpriseOps-Gym retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m225-enterpriseopsgym-current-checkpoint-email-retrieval-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m225_receipt_is_self_hashed_and_source_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["dataset"]["url"] == "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"
    assert payload["dataset"]["revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert payload["dataset"]["oracle"]["records"] == 67
    assert payload["dataset"]["plus_15_tools"]["records"] == 67
    assert payload["dataset"]["tool_execution"] is False
    assert payload["checkpoint"]["parameters"] < 100_000_000


def test_m225_retrieval_gain_is_not_promoted_to_native_success() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]
    assert evaluation["hit_at_1"] == 0.4925373134328358
    assert evaluation["hit_at_3"] == 0.8507462686567164
    assert evaluation["hit_at_5"] == 0.9552238805970149
    assert payload["training_lineage"]["backbone_relative_l2"] == 0.0
    assert payload["comparison"]["warm_minus_baseline_pp"]["hit_at_1"] > 28.0
    assert "do_not_promote" in payload["comparison"]["decision"]
    assert "not an official EnterpriseOps-Gym" in payload["claim_boundary"]
