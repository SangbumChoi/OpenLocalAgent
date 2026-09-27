"""Integrity checks for the current m679 AndroidControl continuation receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m691-m679-androidcontrol-current-v1.json")


def _payload() -> dict:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_m691_is_self_hashed_and_current_checkpoint_bound() -> None:
    payload = _payload()
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert payload["receipt_self_sha256"] == expected
    assert payload["kind"] == "localagent_m691_m679_androidcontrol_current"
    assert payload["source"]["dataset"] == "OfficerChul/Android-Control-84k"
    assert payload["protocol"] == {
        "batch_size": 4,
        "eval_rows": 256,
        "learning_rate": 1.0e-05,
        "max_seq_len": 512,
        "official_native_score": False,
        "split_policy": "pinned public train/eval mirror; no eval rows used for SFT",
        "steps": 32,
        "train_rows": 512,
    }


def test_m691_records_matched_warm_random_control_without_native_claim() -> None:
    payload = _payload()
    assert payload["parent_checkpoint"]["sha256"] == (
        "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    )
    assert payload["comparison"]["warm_after_eval_token_accuracy"] == payload["comparison"]["random_after_eval_token_accuracy"]
    assert payload["comparison"]["exact_sequence_accuracy"] == {"warm": 0.0, "random": 0.0}
    assert payload["environment_executed"] is False
    assert payload["official_split_verified"] is False
    assert payload["weight_adoption"]["warm_action_heads_frozen"] is True
