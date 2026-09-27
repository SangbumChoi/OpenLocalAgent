"""Integrity checks for the current gate with native WebGPU and trajectory evidence."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m419-workshop-gate-current-webgpu-plus-trajectory-v1.json"
)


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m419_gate_is_self_hashed_and_stays_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    checks = {item["requirement"]: item for item in payload["checks"]}
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert any("m407-webgpu-native-current-rerun-v1.json" in item for item in checks["webgpu:native_capability_and_latency"]["evidence"])
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["training:rl_preflight"]["status"] == "pass"
    assert len(payload["blocking_requirements"]) == 10
    companion = payload["companion_evidence"]["m416_local_webgpu_trajectory"]
    assert companion["sha256"] == "b236b1f6543c84e5ff0a8a04f05ac3a21b6160ff80f48e2a954986ee859d9c6a"
    assert companion["pass_at_1"] == 1.0


def test_m419_does_not_promote_local_trajectory_to_official_benchmark() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    blocked = {item["requirement"] for item in payload["blocking_requirements"]}
    assert "native:androidworld" in blocked
    assert "native:agentnet" in blocked
    assert "native:mcpmark" in blocked
    assert "artifacts:public_model_demo_manifest" in blocked
