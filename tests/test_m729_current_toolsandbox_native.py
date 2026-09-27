import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m729-current-toolsandbox-native-v1.json"


def test_m729_toolsandbox_receipt_preserves_single_and_multiturn_boundary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m729_current_toolsandbox_native"
    assert payload["official_split_verified"] is False
    assert payload["single_step"]["task_count"] == 3
    assert payload["single_step"]["success_rate"] == 1.0
    assert payload["interactive"]["task_count"] == 3
    assert payload["interactive"]["success_rate"] == 0.0
    assert payload["decision"]["promote_checkpoint"] is False
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
