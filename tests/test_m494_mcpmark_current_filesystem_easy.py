"""Integrity checks for the exact-current-checkpoint MCPMark filesystem receipt."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m494-mcpmark-current-filesystem-easy-grounded-v1.json")


def _payload() -> dict:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_m494_is_current_checkpoint_bound_and_self_hashed() -> None:
    payload = _payload()
    assert payload["kind"] == "localagent_mcpmark_current_filesystem_easy_receipt"
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 10
    assert payload["success_rate"] == 0.2
    assert payload["checkpoint_sha256"] == payload["model"]["sha256"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m494_records_two_passes_without_official_claim() -> None:
    payload = _payload()
    assert payload["summary"]["verifier_passes"] == 2
    assert payload["summary"]["verifier_failures"] == 8
    assert payload["decision"]["promotion"].startswith("blocked_pending_official")
