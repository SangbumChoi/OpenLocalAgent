"""Integrity checks for the current-checkpoint WebGPU local trajectory receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m450-webgpu-current-local-trajectory-v1.json")
CURRENT_CHECKPOINT_SHA256 = (
    "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
)


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m450_is_self_hashed_and_current_checkpoint_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["environment_executed"] is True
    assert payload["backend"] == "webgpu"
    assert payload["bundle_identity"]["checkpoint_sha256"] == CURRENT_CHECKPOINT_SHA256
    summary = payload["trajectory_result"]["summary"]
    assert summary["steps"] == 13
    assert summary["trajectories"] == 3
    assert summary["schema_valid_rate"] == 1.0
    assert summary["exact_action_rate"] == 1.0
    assert summary["state_transition_rate"] == 1.0
    assert summary["closed_loop_success_rate"] == 1.0
    assert summary["pass_at_1"] == 1.0
    assert all(row["trajectory_success"] for row in summary["by_trajectory"].values())
    assert payload["runner"]["response_status"] == 200
    assert payload["runner"]["page_error_count"] == 0
    assert "No real account" in payload["claim_boundary"]
