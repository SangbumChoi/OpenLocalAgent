import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m180-current-child-mixed-public-continuation-v1.json")


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m180_mixed_continuation_is_source_pinned_and_controlled() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    _self_hash(receipt)
    assert receipt["source"]["rows"] == {"train": 549, "eval": 145}
    assert receipt["source"]["split_policy"].startswith("AgentNet parent records")
    assert receipt["warm_start"]["eval"]["after"]["token_accuracy"] == 0.5805855161787365
    assert receipt["matched_random_control"]["eval"]["after"]["token_accuracy"] == 0.1292758089368259
    comparison = receipt["comparison"]
    assert comparison["warm_start_better_on_held_out_language_model"] is True
    assert comparison["warm_start_better_on_route"] is True
    assert comparison["warm_start_better_on_sequence_exact"] is False
    assert receipt["warm_start"]["eval"]["selector"]["interpretation"] == (
        "not_interpretable_catalog_mismatch"
    )
    assert receipt["matched_random_control"]["eval"]["selector"]["interpretation"] == (
        "not_interpretable_catalog_mismatch"
    )
    assert receipt["comparison"]["decision"].startswith("retain_warm_start")
