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


def test_m201_toolsandbox_interactive_result_is_hash_bound_and_unpromoted() -> None:
    path = Path("docs/paper/results/raw/m201-toolsandbox-native-current-interactive-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source_url"] == "https://github.com/apple/ToolSandbox"
    assert receipt["protocol"]["name"] == "bounded_multi_step_scripted_user"
    assert receipt["protocol"]["interactive"] is True
    assert receipt["protocol"]["user_simulator_executed"] is False
    assert receipt["metrics"]["task_count"] == 5
    assert receipt["metrics"]["success_count"] == 1
    assert receipt["metrics"]["success_rate"] == 0.2
    assert receipt["comparison"]["all_scenarios_match_m92_stateful"] is True
    assert receipt["comparison"]["all_scenarios_match_m92_public"] is True
    assert receipt["comparison"]["all_scenarios_match_m92_projection"] is True
    assert receipt["decision"]["checkpoint_promoted"] is False
