"""Integrity checks for the matched multi-step native AppWorld probe."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m694-m679-appworld-free-v1.json")


def _payload() -> dict:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_m694_is_self_hashed_and_environment_bound() -> None:
    payload = _payload()
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected
    assert payload["environment_contract"]["oracle_contract_passed"] is True
    assert payload["protocol"]["max_steps"] == 4


def test_m694_records_zero_completion_and_no_warm_gain() -> None:
    payload = _payload()
    expected = {name: 0 for name in ("m679_warm", "m679_random", "m692_warm", "m692_random")}
    assert payload["comparison"]["native_task_success"] == expected
    assert payload["comparison"]["native_task_success_rate"] == {name: 0.0 for name in expected}
    assert payload["comparison"]["all_arms_zero_task_completion"] is True
    assert payload["comparison"]["m692_warm_random_action_sequences_identical"] is True
    assert payload["weight_adoption"]["m692_warm_vs_random_free_run_gain_pp"] == 0.0
