import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m651-appworld-trajectory-transfer-v1.json")


def test_m651_trajectory_transfer_preserves_weight_and_native_boundaries() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["train_rows"] == 64
    assert payload["benchmark"]["eval_rows"] == 18
    assert payload["metrics"]["warm_after_minus_random_after_pp"] > 20.0
    assert payload["metrics"]["warm_exact_sequence_accuracy"] == 0.0
    assert payload["decision"]["promote_to_native_appworld_or_webgpu_success"] is False
