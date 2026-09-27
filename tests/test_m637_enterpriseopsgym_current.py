import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m637-m626-enterpriseopsgym-current-email-retrieval-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m637_enterpriseopsgym_receipt_is_current_and_nonofficial() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["dataset_url"] == "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"
    assert payload["benchmark"]["dataset_revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert payload["checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["summary"]["records"] == 67
    assert payload["protocol"]["verifiers_dropped"] is True
