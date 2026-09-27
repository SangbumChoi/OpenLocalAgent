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


def test_m143_toolsandbox_native_base_transfer_is_bound_and_unpromoted() -> None:
    path = Path("docs/paper/results/raw/m143-toolsandbox-native-base-transfer-audit-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source_url"] == "https://github.com/apple/ToolSandbox"
    assert receipt["coverage"] == {
        "augmented_variants_per_base": 8,
        "base_scenarios": 129,
        "selection": "all 129 upstream base/no-distraction scenarios; sorted names; random seed 2028 before named_scenarios",
        "source_level_scenarios": 1032,
    }
    assert receipt["warm"]["environment_executed"] is True
    assert receipt["warm"]["verifier_executed"] is True
    assert receipt["warm"]["official_split_verified"] is False
    assert receipt["warm"]["task_count"] == 129
    assert receipt["warm"]["success_count"] == 28
    assert receipt["random_control"]["task_count"] == 129
    assert receipt["random_control"]["success_count"] == 28
    assert receipt["comparison"]["same_scenario_order"] is True
    assert receipt["comparison"]["warm_minus_random_success_rate"] == 0.0
    assert receipt["decision"]["checkpoint_promoted"] is False
    assert receipt["decision"]["native_official_gate_eligible"] is False
