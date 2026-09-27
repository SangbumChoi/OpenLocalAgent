import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m700-m679-mcpmark-playwright-standard-v1.json")


def test_m700_binds_current_checkpoint_and_real_browser_service() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m700_m679_mcpmark_playwright_standard"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 2
    assert payload["summary"]["runtime_errors"] == 0
    assert payload["summary"]["verifier_passes"] == 0
    assert payload["checkpoint_sha256"] == payload["model"]["sha256"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m700_records_browser_action_but_no_promotion() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["decision"]["browser_action_evidence"]["navigation_and_snapshot_without_server_error"]
    assert payload["decision"]["promotion"] == "blocked_bounded_subset_and_verifier_zero"
