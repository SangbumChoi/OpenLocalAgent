"""Integrity checks for the bounded public xLAM-derived warm/random receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m463-xlam-derived-warm-random-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
REVISION = "dfbd3c669354c27f2727870d39a4d86c32381448"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m463_binds_derivative_source_parent_and_protocol() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_xlam_derived_warm_random_continuation_receipt"
    assert payload["dataset"]["revision"] == REVISION
    assert payload["dataset"]["official_salesforce_split_verified"] is False
    assert payload["dataset"]["train"]["rows"] == 256
    assert payload["dataset"]["eval"]["rows"] == 128
    assert payload["parent_checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["protocol"]["steps"] == 24
    assert payload["protocol"]["random_backbone_init"] == "random"


def test_m463_requires_a_candidate_decision_and_blocks_export() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["warm_eval_sequence_accuracy_after"] == 0.0
    assert comparison["random_eval_sequence_accuracy_after"] == 0.0
    assert payload["decision"]["adoption"] in {
        "retain_as_low_rate_initialization_candidate",
        "reject_warm_initialization_candidate",
    }
    assert payload["decision"]["native_replay_required"] is True
    assert payload["decision"]["webgpu_export_allowed"] is False
