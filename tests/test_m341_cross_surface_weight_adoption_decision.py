import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m341-cross-surface-weight-adoption-decision-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m341_binds_current_parent_and_all_transfer_evidence() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["current_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    receipts = {item["receipt"] for item in payload["evidence"]}
    assert len(receipts) == 7
    assert all((ROOT / path).is_file() for path in receipts)


def test_m341_keeps_export_and_native_promotion_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    decision = payload["decision"]
    assert decision["backbone"] == "retain_as_initialization_candidate_only"
    assert decision["backbone_update_policy"] == (
        "low_rate_only_after_frozen_and_matched_random_controls"
    )
    assert decision["native_promotion"] is False
    assert decision["webgpu_export_of_child"] is False
    assert decision["overall"] == "do_not_promote_from_current_evidence"
