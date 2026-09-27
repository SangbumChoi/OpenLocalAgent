import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m644-mind2web-browsergym-native-canary-v1.json")
BASELINE_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
CHILD_SHA = "3b1737e93fbfbdc6c412d8b9385a885098b280494fb07c0e2bdb8839749f0076"


def test_m644_canary_is_paired_and_does_not_claim_official_success() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["baseline"]["checkpoint"]["sha256"] == BASELINE_SHA
    assert payload["mind2web_child"]["checkpoint"]["sha256"] == CHILD_SHA
    assert payload["benchmark"]["episodes"] == 16
    assert payload["benchmark"]["official_split_verified"] is False
    assert payload["baseline"]["success_count"] == payload["mind2web_child"]["success_count"] == 4
    assert payload["paired_comparison"]["identical_step_traces"] == 16
