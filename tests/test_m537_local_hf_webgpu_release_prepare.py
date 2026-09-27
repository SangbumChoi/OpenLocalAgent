import hashlib
import json
from pathlib import Path


def test_m537_release_prepare_is_current_and_parity_verified() -> None:
    path = Path("docs/paper/results/raw/m537-local-hf-webgpu-release-prepare-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["checkpoint"]["parameters"] == 10_524_544
    assert payload["parity"]["hard_gate"] is True
    assert payload["parity"]["passed"] is True
    assert payload["parity"]["tool_count"] == 63
    assert payload["publication"]["published"] is False
