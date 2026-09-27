import hashlib
import json
from pathlib import Path


def test_m544_grounded_transfer_receipt_is_self_hashed_and_fail_closed() -> None:
    path = Path(
        "docs/paper/results/raw/m544-grounded-mind2web-webgpu-browsergym-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["mind2web"]["after_exact_span"] > payload["mind2web"]["before_exact_span"]
    assert payload["webgpu"]["parity_gate"] is True
    assert payload["browsergym"]["official_split_verified"] is False
    assert payload["decision"]["adoption"] == "reject_for_promotion"
