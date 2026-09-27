"""Integrity checks for the official realistic-agent source/method audit."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m484-realistic-source-method-audit-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m484_binds_official_source_methods_and_claim_boundary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_realistic_source_method_audit_receipt"
    assert len(payload["sources"]) == 11
    ids = {source["id"] for source in payload["sources"]}
    assert {"androidcontrol", "androidworld", "iosworld", "mcpmark"} <= ids
    assert "not a benchmark result" in payload["claim_boundary"]


def test_m484_keeps_training_and_native_admission_separate() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert "official/public train partitions" in payload["admission_rules"]["training"]
    assert "cannot substitute" in payload["admission_rules"]["native"]
    by_id = {source["id"]: source for source in payload["sources"]}
    assert by_id["androidworld"]["train_policy"] == (
        "evaluation-only; never export task templates, generated goals, or gold state"
    )
    assert by_id["mcpmark"]["verifier"].startswith("task-specific verification script")
