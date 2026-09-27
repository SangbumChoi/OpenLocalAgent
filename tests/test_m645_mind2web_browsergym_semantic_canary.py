import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m645-mind2web-browsergym-grounding-canary-v1.json")
BASELINE_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
CHILD_SHA = "3b1737e93fbfbdc6c412d8b9385a885098b280494fb07c0e2bdb8839749f0076"


def test_m645_canary_records_grounding_boundary() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark"]["episodes"] == 16
    assert payload["benchmark"]["official_split_verified"] is False
    assert payload["paired_checkpoint_identity"] == {
        "baseline_sha256": BASELINE_SHA,
        "mind2web_child_sha256": CHILD_SHA,
    }
    semantic = payload["semantic_fallback"]
    coordinate = payload["coordinate_fallback"]
    assert semantic["baseline"]["success_count"] == semantic["mind2web_child"]["success_count"] == 4
    assert semantic["paired_comparison"]["success_delta"] == 0
    assert semantic["paired_comparison"]["identical_step_traces_excluding_wall_ms"] == 16
    assert coordinate["baseline"]["success_count"] == coordinate["mind2web_child"]["success_count"] == 8
    assert coordinate["paired_comparison"]["success_delta"] == 0
    assert coordinate["paired_comparison"]["identical_step_traces_excluding_wall_ms"] == 16
