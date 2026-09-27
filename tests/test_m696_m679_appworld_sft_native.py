"""Integrity checks for the longer AppWorld SFT/native control."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m696-m679-appworld-sft-native-v1.json")


def _payload() -> dict:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_m696_is_self_hashed_and_split_bound() -> None:
    payload = _payload()
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected
    assert payload["source"]["train_tasks"] == 90
    assert payload["source"]["dev_tasks"] == 6
    assert payload["protocol"]["sft_steps"] == 128


def test_m696_retains_warm_token_gain_but_rejects_native_promotion() -> None:
    payload = _payload()
    assert payload["comparison"]["warm_after_eval_token_accuracy"] > payload["comparison"]["warm_before_eval_token_accuracy"]
    assert payload["comparison"]["warm_minus_random_after_pp"] > 0.0
    assert payload["comparison"]["native_successes"] == {"warm": 0, "random": 0}
    assert payload["comparison"]["native_success_rate"] == {"warm": 0.0, "random": 0.0}
    assert payload["weight_adoption"]["warm_action_heads_frozen"] is True
