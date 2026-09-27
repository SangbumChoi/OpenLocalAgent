import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m293_selector_first_control_is_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m293-mobilegym-selector-first-canary-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["configuration"]["selector_first"] is True
    assert receipt["result"]["official_split_verified"] is True
    assert receipt["result"]["passed_tasks"] == 0
    assert receipt["comparison"]["selector_first_delta_success_rate"] == 0.0
    assert receipt["adoption"]["promote_child"] is False
