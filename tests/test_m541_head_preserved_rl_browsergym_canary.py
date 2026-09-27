import hashlib
import json
from pathlib import Path


def test_m541_browsergym_canary_is_pinned_and_not_mislabeled_official() -> None:
    path = Path("docs/paper/results/raw/m541-head-preserved-rl-browsergym-canary-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["environment"]["browsergym_revision"] == (
        "9e779f087de9a65668b6974d11f9ce9816026e96"
    )
    assert payload["environment"]["miniwob_revision"] == (
        "7fd85d71a4b60325c6585396ec4f48377d049838"
    )
    assert payload["result"]["episodes"] == 16
    assert payload["result"]["planned_episodes"] == 240
    assert payload["result"]["official_split_verified"] is False
