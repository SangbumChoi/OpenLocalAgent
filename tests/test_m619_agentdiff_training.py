import hashlib
import json
from pathlib import Path


def test_m619_agentdiff_training_receipt_is_self_consistent_and_test_safe() -> None:
    path = Path("docs/paper/results/raw/m619-agentdiff-training-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["dataset"]["train_rows"] == 179
    assert payload["dataset"]["test_rows"] == 45
    assert payload["dataset"]["test_policy"] == "eval_only"
    assert payload["comparison"]["warm_beats_random_after"] is True
    assert payload["comparison"]["after_test_warm_minus_random_pp"] > 15.0
    assert payload["decision"]["admit_test_rows_to_training"] is False
    assert payload["decision"]["export_to_webgpu"] is False
