import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m638-m626-enterpriseopsgym-warm-random-transfer-v1.json")
WARM_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
RANDOM_SHA = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"


def test_m638_enterpriseopsgym_ablation_is_current_and_matched() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["dataset_revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert payload["warm_checkpoint"]["identity"]["sha256"] == WARM_SHA
    assert payload["random_checkpoint"]["identity"]["sha256"] == RANDOM_SHA
    assert payload["warm_checkpoint"]["summary"]["records"] == 67
    assert payload["warm_minus_random_delta"]["hit_at_1"] > 0.20
    assert payload["protocol"]["same_source_files"] is True
