"""Integrity checks for the acquired Mind2Web continuation and weight audit."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m420-mind2web-public-continuation-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m420_receipt_is_self_hashed_and_binds_public_train_snapshot() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    acquisition = payload["acquisition"]
    assert acquisition["dataset"] == "osunlp/Mind2Web"
    assert acquisition["revision"] == "17ece8eb89862368edc0cc806acee6fca5163474"
    assert acquisition["raw_snapshot"]["bytes"] == 616000823
    assert acquisition["raw_snapshot"]["sha256"] == "c8b622901057bca813a6d171733c41e4fc266c2902a23d63b9094c0add3f8f2c"
    assert acquisition["derived_filter"]["records_inspected"] == 83
    assert acquisition["derived_filter"]["records_rejected"] == 19
    assert acquisition["derived_filter"]["records_retained"] == 64


def test_m420_split_and_transfer_metrics_are_source_disjoint() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    split = payload["normalization"]["source_disjoint_split"]
    assert split["train_parent_tasks"] == 48
    assert split["eval_parent_tasks"] == 16
    assert split["train_path"] == "data/private/mind2web/normalized/m420/train.jsonl"
    assert split["eval_path"] == "data/private/mind2web/normalized/m420/eval.jsonl"
    assert split["parent_overlap"] == 0
    assert split["typed_slot_overlap"] == 0
    metrics = payload["metrics"]
    assert metrics["teacher_forced"]["eval_after_token_accuracy"] > metrics["teacher_forced"]["eval_before_token_accuracy"]
    assert metrics["teacher_forced"]["eval_sequence_accuracy_after"] == 0.0
    assert metrics["heads"]["eval_selector_top1_after"] > 0.84
    movement = payload["weight_movement"]["relative_l2_percent"]
    assert movement["embedding"] < 0.2
    assert movement["attention"] < 0.2
    assert movement["ffn"] < 0.2
    assert movement["dense_selector"] > 90.0
    assert payload["training"]["child_checkpoint"]["path"] == "runs/sft-mind2web-public-continuation-20260805/latest.pt"
    assert payload["provenance_policy"]["official_test_rows_admitted"] == 0
