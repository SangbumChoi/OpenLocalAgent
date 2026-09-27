from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m265-webgpu-native-current-browser-context-v1.json")


def test_m265_current_checkpoint_native_webgpu_receipt_is_hash_bound() -> None:
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
    assert payload["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["bundle"]["checkpoint_sha256"] == payload["checkpoint"]["sha256"]
    assert payload["capability"]["evaluated_cases"] == 3
    assert payload["capability"]["exact_actions"] == 3
    assert payload["capability"]["closed_loop_success"] == 0
    assert payload["capability"]["external_side_effects_executed"] is False
    assert all(case["exact_actions"] == 30 for case in payload["capability"]["cases"])
    assert payload["performance"]["latency_ms_p50"] < 100
    assert payload["performance"]["tokens_per_second_p50"] > 100
    assert payload["protocol"]["requested_provider"] == ["webgpu"]
    assert payload["protocol"]["session_provider_retry"] is False
    assert "explicit intent guards" in payload["claim_boundary"]
