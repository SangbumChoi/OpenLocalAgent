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


def test_m292_mobile_transfer_receipt_is_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m292-mobile-dispatch-transfer-native-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["public_data"]["train"]["rows"] == 4096
    assert receipt["public_data"]["eval"]["rows"] == 904
    assert receipt["static_transfer"]["warm"]["selector_top1"] > receipt["static_transfer"]["random_control"]["selector_top1"]
    assert receipt["native_mobilegym"]["warm_success"] == 0
    assert receipt["adoption"]["promote_warm_child"] is False
