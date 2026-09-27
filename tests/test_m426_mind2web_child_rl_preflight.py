"""Integrity checks for the failed child-bound RL preflight."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m426-mind2web-child-rl-preflight-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m426_binds_child_and_preserves_failed_status() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["status"] == "failed"
    assert payload["checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["data"]["prompt_overlap"] == 0
    assert payload["data"]["row_overlap"] == 0


def test_m426_rejects_zero_tensor_update_despite_reward_diversity() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    rollout = payload["rollout_observability"]
    transition = payload["policy_transition"]
    assert rollout["reward_distribution"]["unique_values"] == 2
    assert rollout["informative_groups"] == 2
    assert transition["nonzero_learning_rate_executed"] is True
    assert transition["changed_model_parameter_count"] == 0
    assert transition["initial_model_state_sha256"] == transition["final_model_state_sha256"]
