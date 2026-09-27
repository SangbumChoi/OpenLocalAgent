import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m381-current-stateful-lowrate-80-v1.json"


def test_m381_lowrate_80_is_current_bound_and_not_promoted() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["kind"] == "localagent_stateful_productivity_transfer_ablation"
    assert payload["parent"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["training"]["train_task_hash"] != payload["training"]["eval_task_hash"]
    assert payload["strict_runtime_replay"]["oracle_task_complete_rate"] == 1.0
    assert payload["strict_runtime_replay"]["model_task_complete_rate"] == 0.2
    assert payload["strict_runtime_replay"]["model_accepted_steps"] == 1
    assert payload["comparison"]["lowrate_minus_frozen_task_complete_rate"] == 0.0
    assert payload["comparison"]["transfer_adoption_decision"] == (
        "do_not_adopt_as_capability_evidence"
    )
