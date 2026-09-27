import hashlib
import json
from pathlib import Path


def test_m617_agentworldbench_transfer_is_self_consistent_and_eval_only() -> None:
    path = Path("docs/paper/results/raw/m617-agentworldbench-transfer-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["dataset"]["rows"] == 224
    assert payload["dataset"]["train_policy"] == "eval_only"
    assert payload["metrics"]["overall"]["warm_token_accuracy"] > payload["metrics"]["overall"]["random_token_accuracy"]
    assert payload["metrics"]["overall"]["warm_minus_random_pp"] > 6.0
    assert payload["metrics"]["overall"]["warm_exact_sequence_accuracy"] == 0.0
    assert payload["decision"]["export_to_webgpu"] is False
    assert payload["decision"]["native_promotion"] is False
