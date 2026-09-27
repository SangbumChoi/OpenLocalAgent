import hashlib
import json
from pathlib import Path


def test_m542_browsergym_grounding_canary_is_self_hashed_and_non_official() -> None:
    path = Path(
        "docs/paper/results/raw/m542-browsergym-realistic-pool-grounding-canary-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["environment"]["tool_pool"] == "realistic_browser"
    assert payload["result"]["episodes"] == 16
    assert payload["result"]["planned_episodes"] == 240
    assert payload["result"]["successful_episodes"] == 4
    assert payload["result"]["official_split_verified"] is False

