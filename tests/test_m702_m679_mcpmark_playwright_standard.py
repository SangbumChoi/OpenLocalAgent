import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m702-m679-mcpmark-playwright-standard-v1.json")


def test_m702_binds_four_task_native_browser_subset() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m702_m679_mcpmark_playwright_standard"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 4
    assert payload["summary"]["runtime_errors"] == 0
    assert payload["summary"]["verifier_passes"] == 0
    assert payload["checkpoint_sha256"] == payload["model"]["sha256"]
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected


def test_m702_records_tool_errors_without_promotion() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["summary"]["browser_tool_errors"] == 2
    assert payload["decision"]["promotion"] == "blocked_bounded_subset_and_verifier_zero"
