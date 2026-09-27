"""Integrity checks for the current-checkpoint local HF/WebGPU release candidate."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m449-hf-webgpu-local-release-v1.json")
CHECKPOINT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m449_local_release_is_current_checkpoint_bound_and_parity_verified() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_hf_webgpu_local_release_receipt"
    assert payload["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert payload["checkpoint"]["parameters"] == 10524544
    assert payload["webgpu_bundle"]["parity_gate_passed"] is True
    assert payload["webgpu_bundle"]["bundle_identity_sha256"] == (
        "ff0259b3f86c08de56533a32bd3db61783a8077e8090cb84e2bca6393258fc00"
    )


def test_m449_does_not_claim_hugging_face_publication_without_authentication() -> None:
    publication = json.loads(RECEIPT.read_text(encoding="utf-8"))["publication"]
    assert publication["published"] is False
    assert publication["hf_authenticated"] is False
    assert publication["model_url"] is None
    assert publication["space_url"] is None
    assert publication["blocker"] == "HF_TOKEN_or_hf_auth_login_required"
