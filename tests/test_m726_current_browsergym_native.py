import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m726-current-browsergym-native-v1.json"


def test_m726_native_receipt_is_current_checkpoint_bound_and_negative() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m726_current_browsergym_native"
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 8
    assert payload["success_count"] == 0
    assert payload["success_rate"] == 0.0
    assert payload["decision"]["native_browser_promotion"] is False
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
