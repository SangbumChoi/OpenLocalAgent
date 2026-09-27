"""Integrity checks for the current-child ToolSandbox native receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m422-toolsandbox-native-child-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m422_binds_child_and_runs_pinned_native_verifier() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "toolsandbox"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["task_count"] == 3
    assert payload["success_count"] == 3
    assert payload["success_rate"] == 1.0


def test_m422_keeps_interactive_failure_and_publication_boundary_explicit() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    interactive = payload["interactive_stress"]
    assert interactive["task_count"] == 1
    assert interactive["success_rate"] == 0.0
    assert interactive["similarity"] == 0.0
    assert "official-split requirement" in payload["claim_boundary"]
