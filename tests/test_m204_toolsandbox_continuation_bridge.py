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


def test_m204_public_continuation_does_not_overclaim_native_adoption() -> None:
    path = Path("docs/paper/results/raw/m204-toolsandbox-continuation-native-bridge-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["dataset"]["name"] == "apple/ToolSandbox"
    assert receipt["dataset"]["train"]["rows"] == 107
    assert receipt["dataset"]["eval"]["rows"] == 20
    assert receipt["training"]["warm"]["eval_token_accuracy_after"] > receipt["training"]["warm"]["eval_token_accuracy_before"]
    assert receipt["training"]["warm"]["eval_sequence_accuracy_after"] == 0.0
    assert receipt["native_interactive"]["per_scenario_parity"] is True
    assert receipt["native_interactive"]["warm_minus_random_success_rate_pp"] == 0.0
    assert receipt["decision"]["export_child"] is False
    assert receipt["decision"]["adopt_warm_backbone"] is False
