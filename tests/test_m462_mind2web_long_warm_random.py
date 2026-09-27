"""Integrity checks for the matched warm/random Mind2Web continuation receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m462-mind2web-long-warm-random-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
DATASET_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m462_binds_same_public_source_parent_and_protocol() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mind2web_long_warm_random_comparison_receipt"
    assert payload["dataset"]["revision"] == DATASET_REVISION
    assert payload["parent_checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["protocol"]["steps"] == 24
    assert payload["protocol"]["learning_rate"] == 1e-5
    assert payload["protocol"]["random_backbone_init"] == "random"


def test_m462_warm_beats_random_but_is_not_promoted() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["warm_wins_teacher_forced_tokens"] is True
    assert comparison["warm_eval_token_accuracy_after"] > comparison[
        "random_eval_token_accuracy_after"
    ]
    assert comparison["warm_eval_sequence_accuracy_after"] == 0.0
    assert comparison["random_eval_sequence_accuracy_after"] == 0.0
    assert payload["decision"]["adoption"] == "retain_as_low_rate_initialization_candidate"
    assert payload["decision"]["native_replay_required"] is True
    assert payload["decision"]["webgpu_export_allowed"] is False
