"""Integrity checks for the local realistic-agent dependency preflight."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m477-realistic-agent-preflight-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m477_preflight_is_fail_closed_and_catalog_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["catalog_entries"] == 42
    assert payload["counts"] == {
        "blocked": 38,
        "evaluation_or_restricted_rows": 38,
        "runnable": 4,
        "train_rows": 4,
    }
    assert payload["runnable_ids"] == [
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    ]
    assert "iosworld" in payload["blocked_ids"]
    assert "mobile_safety_bench" in payload["blocked_ids"]
    assert payload["decision"]["native_evaluation_ready"] is False
    assert payload["decision"]["train_data_admission_unchanged"] is True
