"""Integrity checks for the public safety-source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m226-realistic-safety-source-audit-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m226_sources_are_primary_linked_and_eval_only() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    sources = {source["id"]: source for source in payload["sources"]}
    assert set(sources) == {"vpi_bench", "agentcibench"}
    assert sources["vpi_bench"]["reported_cases"] == 306
    assert sources["vpi_bench"]["training_used"] is False
    assert "Email" in sources["vpi_bench"]["reported_platforms"]
    assert sources["agentcibench"]["reported_seed_scenarios"] == 28
    assert "must_not_share" in sources["agentcibench"]["scenario_contract"]
    assert sources["agentcibench"]["training_used"] is False
    assert "not a downloaded dataset" in payload["claim_boundary"]
