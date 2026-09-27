import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m640-m626-agentnet-current-text-control-v1.json")
WARM_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
RANDOM_SHA = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"


def test_m640_agentnet_control_is_bounded_and_current() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["dataset_url"] == "https://huggingface.co/datasets/xlangai/AgentNet"
    assert payload["warm_full"]["checkpoint"]["sha256"] == WARM_SHA
    assert payload["matched_subset"]["random"]["checkpoint"]["sha256"] == RANDOM_SHA
    assert payload["matched_subset"]["parents"] == 4
    assert payload["matched_subset"]["warm_minus_random"]["first_action_type_rate"] == 0.75
    assert payload["matched_subset"]["warm_minus_random"]["exact_trajectory_rate"] == 0.0
