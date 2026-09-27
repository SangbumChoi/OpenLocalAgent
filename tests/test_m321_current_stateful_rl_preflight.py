import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m321-current-stateful-rl-preflight-v1.json"


def test_m321_current_stateful_rl_preflight_is_passed_and_current_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["kind"] == "localagent_one_update_training_preflight"
    assert payload["status"] == "passed"
    assert payload["metrics"]["lineage"]["parent_checkpoint_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["effective"]["config_payload"]["environment"]["name"] == (
        "stateful_productivity"
    )
    assert payload["measurement"]["rollout_observability"]["reward"]["unique_values"] >= 2
    assert payload["measurement"]["policy_transition"]["changed_model_parameter_count"] > 0
