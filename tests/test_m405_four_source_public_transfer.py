"""Integrity checks for the four-source public transfer receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m405-four-source-public-continuation-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = (
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def test_m405_receipt_is_self_hashed_and_binds_matched_public_sources() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_four_source_public_transfer_receipt"
    assert payload["parent"]["sha256"] == "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    assert payload["parent_checkpoint"] == payload["parent"]
    assert payload["rows"] == {"train": 170, "eval": 80}
    assert payload["comparison"]["aggregate"]["warm_start_better_after"] is True
    assert payload["comparison"]["aggregate"]["warm_minus_random_after_pp"] > 56.0
    assert payload["comparison"]["decision"] == "warm_start_dominates_matched_random_on_all_surfaces"
    assert payload["comparison"]["arm_contract"] == {
        "warm_backbone_init": "parent",
        "random_backbone_init": "random",
        "random_backbone_seed": 2028,
    }
    assert payload["weight_transfer_analysis"]["warm"]["compatibility"]["tokenizer_sha256_equal"] is True
    assert payload["weight_transfer_analysis"]["random"]["compatibility"]["tokenizer_sha256_equal"] is True
    assert {item["label"] for item in payload["sources"]["eval"]} == {
        "androidcontrol",
        "aitw",
        "mind2web",
        "xlam",
    }
    assert "No official benchmark score" in payload["claim_boundary"]
