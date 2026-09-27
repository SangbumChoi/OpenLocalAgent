import hashlib
import json
from pathlib import Path


def test_m546_multisurface_receipt_is_self_hashed_and_claim_bounded() -> None:
    path = Path(
        "docs/paper/results/raw/m546-multisurface-public-transfer-webgpu-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["training"]["after_token_accuracy"] > payload["training"]["before_token_accuracy"]
    assert payload["training"]["after_sequence_accuracy"] == 0.0
    assert payload["weight_transfer"]["shape_mismatches"] == {}
    assert payload["webgpu"]["parity_gate"] is True
    assert payload["webgpu"]["native_capability"]["closed_loop_success"] == 0
    assert payload["decision"]["adoption"] == "retain_as_multisurface_candidate"
