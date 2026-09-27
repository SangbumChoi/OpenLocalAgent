import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m323-current-stateful-long-continuation-v1.json"
WEIGHT = ROOT / "docs/paper/results/raw/m323-current-stateful-long-weight-v1.json"


def test_m323_long_stateful_continuation_records_negative_quality_and_weight_audit() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["parent"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["training"]["sft_steps"] == 64
    assert payload["training"]["rl_steps"] == 8
    assert payload["training"]["reward_unique_values"] == 6
    assert payload["training"]["exact_match_accuracy_post"] < payload["training"]["exact_match_accuracy_pre"]
    assert payload["decision"]["promote_child"] is False
    assert payload["runtime_comparison"]["delta"]["task_complete_rate"] == 0.0

    weight = json.loads(WEIGHT.read_text(encoding="utf-8"))
    assert weight["compatibility"]["config_mismatches"] == {}
    assert weight["compatibility"]["shape_mismatches"] == {}
    assert weight["compatibility"]["tokenizer_sha256_equal"] is True
    assert weight["groups"]["embedding"]["relative_delta_l2"] > 0.05
    assert weight["groups"]["action_heads"]["relative_delta_l2"] == 0.0
