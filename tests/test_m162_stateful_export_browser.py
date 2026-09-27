import json
import hashlib
from pathlib import Path


def test_m162_export_browser_receipt_is_explicitly_diagnostic() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m162-stateful-export-browser-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_webgpu_stateful_current_export_trajectory_receipt"
    assert payload["decision"] == "diagnostic_only"
    assert payload["browser"]["provider_requested"] == "webgpu"
    assert payload["browser"]["observed_backend"] == "WEBGPU"
    assert payload["bundle"]["parity_gate"]["passed"] is True
    assert payload["structured_action_parity"]["passed"] is False
    assert payload["browser"]["summary"]["pass_at_1"] == 0
    assert "not AndroidWorld" in payload["claim_boundary"]
    expected = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "receipt_self_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert payload["receipt_self_sha256"] == expected
