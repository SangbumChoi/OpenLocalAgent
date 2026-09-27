import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m697-m679-mcpmark-filesystem-standard-v1.json")


def test_m697_receipt_is_current_bound_and_self_hashed() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_mcpmark_current_filesystem_standard_receipt"
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is True
    assert payload["task_count"] == 30
    assert payload["success_rate"] == 0.0
    assert payload["checkpoint_sha256"] == payload["model"]["sha256"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m697_records_native_failures_without_promotion() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["summary"]["verifier_passes"] == 0
    assert payload["summary"]["verifier_failures"] == 30
    assert payload["summary"]["runtime_errors"] == 0
    assert payload["decision"]["promotion"].startswith("blocked_pending")
