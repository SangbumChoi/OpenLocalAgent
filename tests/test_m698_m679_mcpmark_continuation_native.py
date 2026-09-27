import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m698-m679-mcpmark-continuation-native-v1.json")


def test_m698_binds_training_and_native_child() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m698_m679_mcpmark_continuation_native"
    assert payload["training"]["train_rows"] == 10
    assert payload["training"]["eval_rows"] == 5
    assert payload["training"]["steps"] == 128
    assert payload["native"]["task_count"] == 30
    assert payload["native"]["runtime_errors"] == 0
    assert payload["native"]["verifier_passes"] == 0
    assert payload["comparison"]["text_gain_transferred_to_native_state"] is False
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m698_does_not_promote_zero_native_child() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["training"]["after_eval"]["assistant_token_accuracy"] > payload["training"]["before_eval"]["assistant_token_accuracy"]
    assert payload["training"]["after_eval"]["assistant_sequence_accuracy"] == 0.0
    assert payload["decision"]["promotion"] == "blocked_native_stateful_execution_zero"
