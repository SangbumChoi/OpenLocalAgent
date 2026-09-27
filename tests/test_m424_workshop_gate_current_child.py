"""Integrity checks for the fail-closed m420-child publication gate."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m424-workshop-gate-current-child-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m424_gate_binds_child_and_remains_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["checks"]["toolsandbox_child_receipt"] == "blocked:official_split_not_verified"
    assert payload["checks"]["mcpmark_child_receipt"] == "blocked:official_split_not_verified"
    assert payload["checks"]["public_model_demo_manifest"] == "blocked:local_candidate_not_public"
    assert "weights:transfer_and_no_transfer_ablation:parent_checkpoint_mismatch" in payload[
        "blocking_requirements"
    ]
