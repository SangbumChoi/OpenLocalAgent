from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m241-appworld-current-checkpoint-native-probe-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def test_m241_appworld_receipt_is_self_hashed_and_native_but_not_promoted() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    assert _canonical_hash(payload) == expected
    assert payload["kind"] == "localagent_appworld_checkpoint_native_probe"
    assert payload["runner"]["package"] == "appworld"
    assert payload["runner"]["contract_verification"]["tasks"] == 1
    assert payload["runner"]["contract_verification"]["passed"] == 1
    assert payload["environment"]["native_runtime_executed"] is True
    assert payload["environment"]["environment_reset_per_task"] is True
    assert payload["environment"]["external_accounts"] is False
    assert payload["summary"] == {
        "action_replayed": 0,
        "native_api_calls": 0,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 6,
    }
    assert all(task["action_replayed"] is False for task in payload["tasks"])
    assert "not an AppWorld leaderboard result" in payload["claim_boundary"]
