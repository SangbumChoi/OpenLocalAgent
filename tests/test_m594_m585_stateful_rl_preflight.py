import hashlib
import json
from pathlib import Path


ROOT = Path("docs/paper/results/raw")
M585_WARM_SHA = "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"


def _load_verified() -> dict:
    payload = json.loads(
        (ROOT / "m594-m585-stateful-rl-preflight-v1.json").read_text(encoding="utf-8")
    )
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    return payload


def test_m594_rl_preflight_is_bound_to_current_warm_checkpoint() -> None:
    payload = _load_verified()
    assert payload["kind"] == "localagent_one_update_training_preflight"
    assert payload["status"] == "passed"
    assert payload["current_checkpoint"]["sha256"] == M585_WARM_SHA
    assert payload["lineage"]["parent_checkpoint_sha256"] == M585_WARM_SHA
    assert payload["protocol"]["train_eval_row_overlap"] == 0
    assert payload["measurement"]["realized_optimizer_updates"] == 2
    assert payload["measurement"]["changed_model_parameter_count"] == 40


def test_m594_weight_transition_favors_attention_and_ffn_over_norms() -> None:
    payload = _load_verified()
    components = payload["weight_transition"]["by_component"]
    assert payload["weight_transition"]["all_named_policy_tensors_changed"] is True
    assert components["ffn"]["relative_l2"] > components["norm"]["relative_l2"]
    assert components["attention"]["relative_l2"] > components["norm"]["relative_l2"]
    assert payload["measurement"]["exact_success_rollouts"] == 0
