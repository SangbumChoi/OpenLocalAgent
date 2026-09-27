import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m705-m679-browser-transfer-native-v1.json")


def test_m705_binds_browser_sft_and_native_child() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m705_m679_browser_transfer_native"
    assert payload["training"]["train_rows"] == 2
    assert payload["training"]["eval_rows"] == 1
    assert payload["training"]["steps"] == 256
    assert payload["native"]["task_count"] == 4
    assert payload["native"]["verifier_passes"] == 0
    assert payload["native"]["runtime_errors"] == 0
    assert payload["comparison"]["text_gain_transferred_to_native_browser"] is False
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m705_reports_positive_text_gain_but_blocks_promotion() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["comparison"]["held_out_token_accuracy_gain_pp"] > 20.0
    assert payload["training"]["after_eval"]["assistant_sequence_accuracy"] == 0.0
    assert payload["decision"]["promotion"] == "blocked_native_browser_verifier_zero"
