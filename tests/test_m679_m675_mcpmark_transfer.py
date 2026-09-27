import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m679_mcpmark_transfer_binds_redacted_public_split() -> None:
    path = ROOT / "docs/paper/results/raw/m679-m675-mcpmark-transfer-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["source"]["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert payload["protocol"]["train_rows"] == 10
    assert payload["protocol"]["eval_rows"] == 5
    assert payload["protocol"]["official_native_score"] is False
    assert payload["comparison"]["warm_minus_random_after_pp"] > 11.0
    assert payload["comparison"]["exact_sequence_accuracy"] == {"warm": 0.0, "random": 0.0}
    assert payload["weight_adoption"]["reuse_warm_backbone"] is True
