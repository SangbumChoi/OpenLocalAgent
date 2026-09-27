import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m305-public-hf-legacy-current-audit-v1.json")


def test_m305_receipt_is_self_hashed_and_rejects_legacy_public_release() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = {
        key: value for key, value in payload.items() if key != "receipt_self_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert payload["receipt_self_sha256"] == digest
    assert payload["verification"]["public_model_http_status"] == 200
    assert payload["verification"]["public_demo_http_status"] == 200
    assert payload["verification"]["current_checkpoint_match"] is False
    assert payload["current_checkpoint_sha256"] is None
    assert payload["verification"]["local_checkpoint_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
