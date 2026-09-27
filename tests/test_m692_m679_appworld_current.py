"""Integrity checks for current m679 AppWorld public trajectory continuation."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m692-m679-appworld-current-v1.json")


def _payload() -> dict:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_m692_is_self_hashed_and_split_bound() -> None:
    payload = _payload()
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert payload["receipt_self_sha256"] == expected
    assert payload["source"]["train_tasks"] == 90
    assert payload["source"]["dev_tasks"] == 6
    assert payload["protocol"]["observation_policy"].startswith("bounded redacted")


def test_m692_records_current_parent_and_no_native_claim() -> None:
    payload = _payload()
    assert payload["parent_checkpoint"]["sha256"] == (
        "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    )
    assert payload["comparison"]["warm_after_eval_token_accuracy"] == payload["comparison"]["random_after_eval_token_accuracy"]
    assert payload["comparison"]["exact_sequence_accuracy"] == {"warm": 0.0, "random": 0.0}
    assert payload["environment_executed"] is False
    assert payload["official_split_verified"] is False
