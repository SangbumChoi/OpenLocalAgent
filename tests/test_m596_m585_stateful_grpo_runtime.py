import hashlib
import json
from pathlib import Path


ROOT = Path("docs/paper/results/raw")
WARM_SHA = "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"


def _load() -> dict:
    payload = json.loads(
        (ROOT / "m596-m585-stateful-grpo-runtime-v1.json").read_text(encoding="utf-8")
    )
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    return payload


def test_m596_binds_actual_grpo_child_and_runtime_result() -> None:
    payload = _load()
    assert payload["status"] == "measured"
    assert payload["checkpoint"]["parent_warm_sha256"] == WARM_SHA
    assert payload["training"]["sft_updates"] == 32
    assert payload["training"]["rl_updates"] == 8
    assert payload["runtime"]["oracle_task_complete_rate"] == 1.0
    assert payload["runtime"]["model_task_complete_rate"] == 0.2


def test_m596_runtime_and_weight_analysis_are_fail_closed() -> None:
    payload = _load()
    assert payload["runtime"]["by_family"]["email"]["task_complete_rate"] == 1.0
    assert payload["runtime"]["by_family"]["browser"]["task_complete_rate"] == 0.0
    assert payload["runtime"]["external_accounts_used"] is False
    assert payload["weight_transition"]["action_head_relative_l2"] == 0.0
    assert payload["weight_transition"]["by_component"]["ffn"]["relative_l2"] > payload[
        "weight_transition"
    ]["by_component"]["norm"]["relative_l2"]
