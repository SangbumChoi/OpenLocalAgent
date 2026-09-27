import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m326-current-stateful-lowrate-transfer-v1.json"


def test_m326_lowrate_transfer_is_current_bound_and_not_promoted() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["kind"] == "localagent_stateful_productivity_transfer_ablation"
    assert payload["parent"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["source"]["train_task_hash"] != payload["source"]["eval_task_hash"]
    lowrate = payload["arms"]["pretrained_lowrate_unfrozen_backbone"]
    assert lowrate["closed_loop"]["task_complete_rate"] == 0.2
    assert lowrate["closed_loop"]["mean_shaped_reward"] > 0.23
    assert lowrate["closed_loop"]["exact_tool_rate"] == 0.375
    assert lowrate["weight_movement"]["backbone"] < 0.002
    assert payload["comparison"]["lowrate_minus_frozen_closed_loop"] == 0.0
    assert payload["comparison"]["transfer_adoption_decision"] == (
        "do_not_adopt_as_capability_evidence"
    )
