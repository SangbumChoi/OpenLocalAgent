from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m271-workshop-gate-tokenizer-bound-v1.json")


def test_m271_gate_uses_corrected_webgpu_receipt_and_stays_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["ready"] is False
    assert payload["inputs"]["webgpu_receipt"].endswith(
        "m270-webgpu-native-tokenizer-bound-v1.json"
    )
    assert payload["webgpu_summary"]["tokenizer_sha256"] == (
        "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c"
    )
    assert payload["webgpu_summary"]["exact_actions"] == 3
    assert payload["webgpu_summary"]["closed_loop_success"] == 0
    assert len(payload["blocked_requirements"]) == 9
