from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m270-webgpu-native-tokenizer-bound-v1.json")


def test_m270_native_webgpu_receipt_binds_checkpoint_tokenizer() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["backend"] == "webgpu"
    assert payload["environment_executed"] is True
    assert payload["checkpoint"]["sha256"] == payload["bundle"]["checkpoint_sha256"]
    assert payload["tokenizer"]["checkpoint_and_bundle_match"] is True
    assert payload["tokenizer"]["sha256"] == payload["tokenizer"]["checkpoint_recorded_sha256"]
    assert payload["capability"]["evaluated_cases"] == 3
    assert payload["capability"]["exact_actions"] == 3
    assert payload["capability"]["closed_loop_success"] == 0
    assert payload["capability"]["external_side_effects_executed"] is False
    assert all(case["exact_actions"] == 30 for case in payload["capability"]["cases"])
    assert payload["protocol"]["requested_provider"] == ["webgpu"]
    assert payload["protocol"]["session_provider_retry"] is False
    assert "checkpoint-recorded BPE tokenizer" in payload["claim_boundary"]
