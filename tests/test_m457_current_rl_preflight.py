"""Current-checkpoint RL preflight and gate integrity checks."""

import hashlib
import json
from pathlib import Path


RL_RECEIPT = Path("docs/paper/results/raw/m457-current-checkpoint-rl-preflight-v1.json")
GATE_RECEIPT = Path("docs/paper/results/raw/m459-workshop-gate-current-rl-pass-v1.json")
CURRENT_CHECKPOINT = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m457_is_current_bound_and_passed_with_policy_transition() -> None:
    payload = json.loads(RL_RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_one_update_training_preflight"
    assert payload["status"] == "passed"
    assert payload["source"]["sft_parent_checkpoint"]["sha256"] == CURRENT_CHECKPOINT
    transition = payload["measurement"]["policy_transition"]
    assert transition["nonzero_learning_rate_executed"] is True
    assert transition["at_least_one_policy_tensor_changed"] is True
    assert transition["changed_model_parameter_count"] > 0
    assert payload["validation_errors"] == []


def test_m459_gate_accepts_rl_but_remains_fail_closed_for_public_release() -> None:
    payload = json.loads(GATE_RECEIPT.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    checks = {item["requirement"]: item for item in payload["checks"]}
    assert checks["training:rl_preflight"]["status"] == "pass"
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["artifacts:public_model_demo_manifest"]["status"] == "blocked"
    assert "manifest_not_supplied" in checks["artifacts:public_model_demo_manifest"]["blockers"]
