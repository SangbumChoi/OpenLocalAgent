import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m192-current-workspace-webgpu-deploy-verification-v1.json"
)


def test_m192_current_workspace_bundle_is_complete_and_hash_verified() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["verified"] is True
    assert payload["deployment_bundle_present"] is True
    assert payload["static_files"] == ["index.html", "app.js", "style.css", "tokenizer.js"]
    assert payload["bundle_artifacts"]["count"] == 8
    assert payload["bundle_artifacts"]["all_hashes_and_byte_counts_match"] is True
    assert payload["bundle_artifacts"]["parity_gate_passed"] is True
    assert payload["model"]["parameters"] == 10524544
    assert "public Hugging Face/Spaces publication" in payload["claim_boundary"]
