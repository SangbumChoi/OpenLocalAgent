import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m184-current-m180-webgpu-export-v1.json")


def test_m184_current_webgpu_export_is_parity_gated_and_local_only() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["parameters"] == 10524544
    assert payload["bundle"]["parity_gate"]["hard_gate"] is True
    assert payload["bundle"]["parity_gate"]["passed"] is True
    assert payload["bundle"]["parity_gate"]["fp16_model_argmax_agreement"] == 1.0
    assert payload["bundle"]["deployment_verification"] == {
        "verified": True,
        "static_app_present": True,
        "blockers": [],
    }
    assert payload["publication"]["uploaded"] is False
