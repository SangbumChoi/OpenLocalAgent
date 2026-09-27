from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m157-mind2web-grounded-transfer-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m157_receipt_self_hash_and_public_boundary() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    assert payload["dataset"] == "osunlp/Mind2Web"
    assert payload["source_revision"] == "17ece8eb89862368edc0cc806acee6fca5163474"
    assert "not the official Mind2Web test score" in payload["claim_boundary"]


def test_m157_is_disjoint_and_warm_pointer_transfer_beats_random_diagnostic() -> None:
    payload = _load()
    split = payload["split_audit"]
    assert split["train_parent_records"] == 9
    assert split["eval_parent_records"] == 3
    assert split["train_decisions"] == 219
    assert split["eval_decisions"] == 63
    assert split["task_id_disjoint"] is True
    assert split["typed_slot_disjoint"] is True
    assert payload["training"]["warm"]["before"]["exact_span"] == 0.0
    assert payload["training"]["random"]["before"]["exact_span"] == 0.0
    assert payload["comparison"]["warm_minus_random_exact_span"] > 0.0
    assert payload["decision"] == "diagnostic_only"


def test_m157_records_expected_pointer_vocab_expansion() -> None:
    payload = _load()
    assert "intentional BROWSER_PTR_ARGS vocabulary expansion" in payload["compatibility_policy"]
    for arm in ("warm", "random"):
        compatibility = payload["training"][arm]["weight_transfer"]["compatibility"]
        assert compatibility["tokenizer_sha256_equal"] is True
        assert compatibility["config_mismatches"] == {}
        assert compatibility["shape_mismatches"]["ptr_head.arg_emb.weight"] == {
            "base": [17, 384],
            "target": [19, 384],
        }
