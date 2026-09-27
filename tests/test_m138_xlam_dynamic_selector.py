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


def test_m138_dynamic_selector_receipt_is_hash_bound_and_not_promoted() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m138-xlam-dynamic-selector-transfer-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["kind"] == "localagent_xlam_dynamic_selector_transfer"
    assert receipt["rows"] == {"eval": 500, "train": 3911, "train_decisions": 3911}
    assert receipt["tools"]["train_union"] == 2702
    assert receipt["tools"]["candidate_union"] == 2840
    assert receipt["source"]["official_salesforce_split_verified"] is False
    assert receipt["metrics"]["after_eval"]["row_local_tool_top1"] > receipt["metrics"]["before_eval"]["row_local_tool_top1"]
    assert receipt["metrics"]["after_eval"]["global_tool_top1"] == 0.02
    assert receipt["decision"]["production_checkpoint_promoted"] is False
    assert receipt["decision"]["workshop_gate_eligible"] is False
