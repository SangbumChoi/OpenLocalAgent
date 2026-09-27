import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m337-mobilegym-current-catalog-reconciliation-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m337_reconciles_current_mobilegym_source_and_native_receipt() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert payload["source"]["official_test_tasks"] == 256
    native = payload["native_evidence"]
    assert native["official_split_verified"] is True
    assert native["native_receipt_eligible"] is True
    assert native["full_official_test_split"] is True
    assert native["passed_tasks"] == 1
    assert native["failed_tasks"] == 255
    assert native["success_rate"] == 1 / 256
    assert native["vision_used"] is False
    assert payload["admission"]["training_rows_admitted"] == 0


def test_m337_keeps_visual_and_training_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["admission"]["benchmark_payload_retained"] is False
    assert "not a visual mobile-agent score" in payload["claim_boundary"]
    assert payload["admission"]["registry_status"] == "measured_official_text_projection"
