import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m643-m626-mind2web-browser-continuation-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m643_mind2web_continuation_is_current_and_source_disjoint() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["dataset"] == "osunlp/Mind2Web"
    assert payload["parent_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["rows"] == {"eval": 32, "train": 96}
    assert payload["split_contract"]["validated_by_training_runner"] is True
    assert payload["comparison"]["aggregate"]["warm_minus_random_after_pp"] > 80.0
    assert payload["decision"]["export_to_webgpu_as_native_browser_agent"] is False
