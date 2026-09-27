import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m647-appworld-current-transfer-v1.json")
WARM_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
RANDOM_SHA = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"


def test_m647_appworld_transfer_is_matched_and_honest() -> None:
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
    assert payload["benchmark"]["protected_test_used"] is False
    assert payload["parent_checkpoints"]["warm"]["sha256"] == WARM_SHA
    assert payload["parent_checkpoints"]["random"]["sha256"] == RANDOM_SHA
    assert payload["metrics"]["warm_after_minus_random_after_token_accuracy_pp"] > 30.0
    assert payload["decision"]["retain_warm_initialization"] is True
    assert payload["decision"]["promote_to_native_appworld_or_webgpu_success"] is False
