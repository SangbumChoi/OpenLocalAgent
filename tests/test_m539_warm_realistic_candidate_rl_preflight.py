import hashlib
import json
from pathlib import Path


def test_m539_warm_candidate_rl_preflight_is_self_hashed_and_passed() -> None:
    path = Path("docs/paper/results/raw/m539-warm-realistic-candidate-rl-preflight-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["policy_transition"]["status"] == "passed"
    assert payload["policy_transition"]["realized_optimizer_updates"] == 2
    assert payload["policy_transition"]["changed_model_parameter_count"] == 40
    assert payload["heldout"]["mean_reward_after"] > payload["heldout"]["mean_reward_before"]
    assert payload["data"]["prompt_overlap"] == payload["data"]["row_overlap"] == 0
