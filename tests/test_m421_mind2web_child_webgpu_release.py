"""Integrity checks for the local WebGPU/HF preparation of the m420 child."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m421-mind2web-child-webgpu-release-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m421_release_receipt_is_self_hashed_and_checkpoint_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    checkpoint = payload["checkpoint"]
    assert checkpoint["parameters"] == 10524544
    assert checkpoint["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    manifest = payload["local_space_bundle"]["manifest"]
    assert manifest["checkpoint_sha256"] == checkpoint["sha256"]
    assert payload["local_space_bundle"]["parity_gate"]["passed"] is True


def test_m421_native_webgpu_fixture_is_complete_but_not_official() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    trajectory = payload["native_webgpu_trajectory"]
    assert trajectory["backend"] == "webgpu"
    assert trajectory["environment_executed"] is True
    assert trajectory["steps"] == 13
    assert trajectory["pass_at_1"] == 1.0
    assert trajectory["page_errors"] == 0
    assert payload["publication_status"]["model_uploaded"] is False
    assert payload["publication_status"]["space_uploaded"] is False
    assert payload["publication_status"]["official_benchmark_score"] is None
