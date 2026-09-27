"""Integrity checks for the fresh current-checkpoint native WebGPU rerun."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m407-webgpu-native-current-rerun-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m407_is_self_hashed_and_checkpoint_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["backend"] == "webgpu"
    assert payload["environment_executed"] is True
    assert payload["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["bundle_identity"]["checkpoint_sha256"] == payload["checkpoint"]["sha256"]
    assert payload["capability"]["evaluated_cases"] == 3
    assert payload["capability"]["exact_actions"] == 3
    assert payload["capability"]["closed_loop_success"] == 0
    assert payload["capability"]["external_side_effects_executed"] is False
    assert payload["performance"]["tokens_per_second_p50"] > 500.0
    assert payload["performance"]["latency_ms_p50"] < 25.0
