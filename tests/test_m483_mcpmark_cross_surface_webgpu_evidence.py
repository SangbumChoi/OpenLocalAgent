"""Integrity checks for the m482 native/WebGPU evidence receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m483-mcpmark-cross-surface-webgpu-evidence-v1.json")
WARM_SHA256 = "4c2355f54194ee38423df148122673d1d287246e714c329911d8b7d6fbf2f813"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m483_binds_child_native_replay_and_webgpu_bundle() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mcpmark_cross_surface_webgpu_evidence_receipt"
    assert payload["checkpoint"]["sha256"] == WARM_SHA256
    assert payload["webgpu"]["parity_gate_passed"] is True
    assert payload["webgpu"]["backend"] == "webgpu"
    assert payload["webgpu"]["hardware_adapter"] == "vendor=apple; architecture=metal-3"


def test_m483_keeps_native_task_and_publication_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["native_mcpmark"]["summary"]["verifier_passes"] == 0
    assert payload["native_mcpmark"]["official_split_verified"] is False
    assert payload["publication"]["published"] is False
    assert payload["publication"]["hf_authenticated"] is False
