"""Integrity checks for the current public EnterpriseOps-Gym email retrieval audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m400-enterpriseopsgym-current-email-retrieval-v1.json"
)
ALIAS_RECEIPT = Path(
    "docs/paper/results/raw/m400-enterpriseopsgym-source-alias-audit-v1.json"
)


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m400_receipt_is_current_checkpoint_bound_and_self_hashed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["dataset"] == "ServiceNow-AI/EnterpriseOps-Gym"
    assert payload["dataset_revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["protocol"]["verifiers_dropped"] is True
    assert payload["protocol"]["server_configuration_dropped"] is True
    assert payload["summary"]["records"] == 67
    assert payload["summary"]["hit_at_1"] == 0.208955223880597
    assert "not an official EnterpriseOps-Gym" in payload["claim_boundary"]


def test_m400_receipt_does_not_publish_prompt_or_verifier_payloads() -> None:
    payload = RECEIPT.read_text(encoding="utf-8")
    assert '"gym_servers_config":' not in payload
    assert '"verifiers":' not in payload
    assert "x-email-user-token" not in payload


def test_m400_alias_audit_keeps_the_rehost_separate_and_self_hashed() -> None:
    payload = json.loads(ALIAS_RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["canonical_source"]["dataset"] == "ServiceNow-AI/EnterpriseOps-Gym"
    assert payload["separate_public_mirror"]["dataset"] == "EnterpriseAgents/EnterpriseOpsGym"
    assert payload["separate_public_mirror"]["rows"] == 649
    assert payload["decision"]["same_source_assumed"] is False
    assert payload["decision"]["use_in_current_receipt"] is False
