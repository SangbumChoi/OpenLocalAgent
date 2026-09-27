import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m633-m626-stateful-rl-preflight-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m633_rl_receipt_is_current_and_self_hashed() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["kind"] == "localagent_one_update_training_preflight"
    assert payload["status"] == "passed"
    assert payload["lineage"]["parent_checkpoint_sha256"] == CURRENT_SHA
    assert payload["protocol"]["train_eval_row_overlap"] == 0
    assert payload["measurement"]["changed_model_parameter_count"] == 40
