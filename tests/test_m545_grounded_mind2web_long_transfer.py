import hashlib
import json
from pathlib import Path


def test_m545_long_dom_transfer_receipt_is_self_hashed_and_not_promoted() -> None:
    path = Path(
        "docs/paper/results/raw/m545-grounded-mind2web-128step-transfer-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["training"]["after_exact_span"] == 80 / 186
    assert payload["webgpu"]["parity_gate"] is True
    assert payload["browsergym"]["success_rate"] == 0.25
    assert payload["browsergym"]["official_split_verified"] is False
    assert payload["decision"]["adoption"] == "reject_for_promotion"
