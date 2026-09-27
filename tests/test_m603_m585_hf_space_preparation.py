import hashlib
import json
from pathlib import Path


def test_m603_release_preparation_is_checkpoint_bound_but_not_published() -> None:
    path = Path("docs/paper/results/raw/m603-m585-hf-space-preparation-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["prepared"] is True
    assert payload["published"] is False
    assert payload["checkpoint"]["sha256"].startswith("6553dc2b")
    assert payload["webgpu_bundle"]["parity_passed"] is True
    assert payload["webgpu_bundle"]["tokenizer_sha256"] == payload["model_bundle"]["tokenizer_sha256"]
    assert "HF_TOKEN" in payload["publication_blocker"]
