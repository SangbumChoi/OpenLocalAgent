import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m211_multisurface_transfer_requires_native_improvement_for_adoption() -> None:
    path = Path("docs/paper/results/raw/m211-multisurface-continuation-native-bridge-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["training"]["warm"]["rows"] == {"train": 4752, "eval": 1069}
    assert receipt["training"]["warm"]["aggregate_eval_token_accuracy"]["after"] > receipt["training"]["warm"]["aggregate_eval_token_accuracy"]["before"]
    assert receipt["training"]["warm"]["eval_sequence_accuracy_after"] == 0.0
    assert receipt["training"]["warm"]["weight_movement_relative_l2"]["action_heads"] == 0.0
    assert receipt["native_toolsandbox_interactive"]["warm_minus_random_success_rate_pp"] == 0.0
    assert receipt["native_toolsandbox_interactive"]["native_success_parity"] is True
    assert receipt["decision"]["export_child"] is False
    assert receipt["decision"]["adopt_warm_backbone"] is False
