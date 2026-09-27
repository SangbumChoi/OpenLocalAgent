import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m673_preparation_is_checkpoint_bound_and_not_published() -> None:
    path = ROOT / "docs/paper/results/raw/m673-m671-hf-space-preparation-v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["prepared"] is True
    assert payload["published"] is False
    assert payload["checkpoint"]["sha256"].startswith("b5576dc8")
    assert payload["model_bundle"]["artifacts"]["model.safetensors"]["bytes"] > 40_000_000
    assert payload["webgpu_bundle"]["parity_passed"] is True
    assert payload["webgpu_bundle"]["bundle_identity_sha256"]
    assert payload["space_staging"]["bundle_manifest_checkpoint_match"] is True
    assert "HF authentication" in payload["publication_blocker"]
