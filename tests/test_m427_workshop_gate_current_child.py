"""Integrity checks for the m427 child-bound fail-closed gate."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m427-workshop-gate-current-child-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m427_binds_child_transfer_and_rejects_rl_publication() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["checks"]["weights_transfer_and_random_ablation"] == "pass"
    assert payload["checks"]["rl_preflight_current_child"] == "blocked:preflight_status_not_passed"
    assert payload["checks"]["public_model_demo_manifest"] == "blocked:local_candidate_not_public"
    assert "native:toolsandbox:official_split_not_verified" in payload["blocking_requirements"]
