"""Integrity checks for the gate recomputed from the fresh WebGPU rerun."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m409-workshop-gate-current-webgpu-rerun-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m409_gate_is_self_hashed_and_uses_fresh_webgpu_receipt() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    checks = {item["requirement"]: item for item in payload["checks"]}
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert any("m407-webgpu-native-current-rerun-v1.json" in item for item in checks["webgpu:native_capability_and_latency"]["evidence"])
    assert len(payload["blocking_requirements"]) == 10
