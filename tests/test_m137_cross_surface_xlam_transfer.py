import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m137_four_source_transfer_receipt_is_hash_bound_and_honest() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m137-cross-surface-xlam-derivative-transfer-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["kind"] == "localagent_cross_surface_xlam_derivative_transfer"
    assert receipt["training"]["train_rows"] == 8540
    assert receipt["training"]["eval_rows"] == 1541
    assert receipt["sources"]["tools"]["official_salesforce_split_verified"] is False
    assert receipt["sources"]["tools"]["train_rows_rejected"] == 89
    assert receipt["sources"]["tools"]["held_out_rows_rejected"] == 18
    assert receipt["sources"]["tools"]["slot_overlap_rows_removed"] == 482
    warm = receipt["arms"]["warm_parent_backbone"]
    random = receipt["arms"]["random_backbone_control"]
    assert warm["after_aggregate_token_accuracy"] > warm["before_aggregate_token_accuracy"]
    assert warm["after_aggregate_token_accuracy"] > random["after_aggregate_token_accuracy"]
    assert warm["weight_relative_delta_l2"]["action_heads"] == 0.0
    assert receipt["xlam_first_call_probe"]["warm_child"]["row_retriever_argument_exact"] == 0.0
    assert receipt["decision"]["production_checkpoint_promoted"] is False
