"""Integrity checks for the warm-child local HF/WebGPU release candidate."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m481-warm-child-hf-webgpu-release-candidate-v1.json")
WARM_SHA256 = "1fc2d4014e746a4f2784338577794a759279fbb3636bdf0e8e3954c4fe6f40db"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m481_binds_local_hf_and_webgpu_artifacts() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_warm_child_hf_webgpu_release_candidate"
    assert payload["checkpoint"]["sha256"] == WARM_SHA256
    assert payload["checkpoint"]["parameters"] == 10_524_544
    assert payload["webgpu_bundle"]["parity_gate_passed"] is True
    assert payload["native_webgpu"]["environment_executed"] is True
    assert payload["native_webgpu"]["backend"] == "webgpu"


def test_m481_keeps_publication_and_productivity_claims_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["publication"]["published"] is False
    assert payload["publication"]["hf_authenticated"] is False
    assert payload["native_mcpmark"]["official_split_verified"] is False
    assert payload["native_mcpmark"]["warm_verifier_pass"] is False
    assert payload["native_webgpu"]["closed_loop_success"] == 0
