import hashlib
import json
from pathlib import Path


def test_m535_toolsandbox_smoke_is_current_and_self_hashed() -> None:
    path = Path("docs/paper/results/raw/m535-toolsandbox-native-current-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["benchmark_id"] == "toolsandbox"
    assert payload["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert payload["source_revision"] == "165848b9a78cead7ca7fe7c89c688b58e6501219"
    assert payload["success_count"] == payload["task_count"] == 3
    assert payload["official_split_verified"] is False
    assert payload["external_api_called"] is False
