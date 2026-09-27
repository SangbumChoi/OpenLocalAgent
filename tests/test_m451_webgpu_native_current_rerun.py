"""Integrity checks for the current-checkpoint native WebGPU capability receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m451-webgpu-native-current-rerun-v1.json")
CURRENT_CHECKPOINT_SHA256 = (
    "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
)


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m451_is_self_hashed_and_current_checkpoint_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["environment_executed"] is True
    assert payload["backend"] == "webgpu"
    assert payload["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert payload["bundle_identity"]["checkpoint_sha256"] == CURRENT_CHECKPOINT_SHA256
    capability = payload["capability"]
    assert capability["evaluated_cases"] == 3
    assert capability["exact_actions"] == 3
    assert capability["closed_loop_success"] == 0
    assert capability["external_side_effects_executed"] is False
    performance = payload["performance"]
    assert performance["tokens_per_second_p50"] > 0
    assert performance["latency_ms_p50"] > 0
    assert performance["peak_memory_mb"] > 0
    assert payload["runner"]["response_status"] == 200
    assert payload["runner"]["page_error_count"] == 0
    assert "no real email" in payload["claim_boundary"].lower()
