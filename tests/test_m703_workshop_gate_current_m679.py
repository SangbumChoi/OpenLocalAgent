import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m703-workshop-gate-current-m679-v1.json")


def test_m703_is_fail_closed_after_full_bounded_browser_subset() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m703_workshop_gate_current_m679"
    assert payload["ready"] is False
    assert payload["blocked_requirements"]["native:mcpmark_playwright"] == "bounded_subset_and_verifier_zero"
    assert payload["current_checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected
