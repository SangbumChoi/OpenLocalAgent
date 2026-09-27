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


def test_m297_toolsandbox_native_receipt_preserves_claim_boundary() -> None:
    path = Path("docs/paper/results/raw/m297-toolsandbox-current-native-smoke-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["runner"]["simulator_executed"] is True
    assert receipt["runner"]["verifier_executed"] is True
    assert receipt["benchmark"]["official_split_verified"] is False
    assert receipt["single_step"]["success_count"] == 2
    assert receipt["interactive"]["success_count"] == 0
    assert receipt["decision"]["promote_checkpoint"] is False
