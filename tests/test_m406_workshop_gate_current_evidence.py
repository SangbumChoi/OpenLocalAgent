"""Integrity checks for the current checkpoint-bound workshop gate refresh."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m406-workshop-gate-current-evidence-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m406_gate_is_self_hashed_and_binds_current_evidence() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    checks = {item["requirement"]: item for item in payload["checks"]}
    for requirement in (
        "native:mobilegym",
        "native:browsergym_miniwob",
        "webgpu:native_capability_and_latency",
        "weights:transfer_and_no_transfer_ablation",
        "training:rl_preflight",
    ):
        assert checks[requirement]["status"] == "pass"
    assert checks["artifacts:public_model_demo_manifest"]["status"] == "blocked"
    assert checks["native:androidworld"]["status"] == "blocked"
    assert len(payload["blocking_requirements"]) == 10
