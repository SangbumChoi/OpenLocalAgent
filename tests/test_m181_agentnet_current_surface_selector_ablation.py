import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m181-agentnet-current-surface-selector-ablation-v1.json")


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m181_agentnet_selector_ablation_is_multiseed_and_unpromoted() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    assert receipt["source"]["official_split_verified"] is False
    assert receipt["source"]["images_consumed"] is False
    assert receipt["selector"]["tool_count"] == 14
    aggregate = receipt["multiseed"]["aggregate"]
    assert aggregate["warm_eval_top1_mean"] == 0.6967418591181437
    assert aggregate["random_eval_top1_mean"] == 0.7067669034004211
    assert aggregate["warm_minus_random_eval_top1_mean"] < 0
    assert aggregate["warm_better_seed_count"] == 0
    replay = receipt["end_to_end_replay"]
    assert replay["completeness_verified"] is True
    assert replay["first_action_type_rate"] == 1.0
    assert replay["success_rate"] == 0.0
    assert replay["exact_trajectory_rate"] == 0.0
    assert receipt["decision"] == "do_not_adopt_surface_selector_as_transfer_evidence"
