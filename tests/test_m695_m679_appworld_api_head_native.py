"""Integrity checks for the AppWorld API-head native control."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m695-m679-appworld-api-head-native-v1.json")


def test_m695_self_hash_and_native_negative() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected
    assert payload["comparison"]["head_eval_exact"] == {"warm": 2, "random": 0}
    assert payload["comparison"]["native_successes"] == {"warm": 0, "random": 0}
    assert payload["weight_adoption"]["backbone_frozen"] is True
