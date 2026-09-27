import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m675_androidcontrol_transfer_binds_public_split_and_warm_gain() -> None:
    path = ROOT / "docs/paper/results/raw/m675-m671-androidcontrol-transfer-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["source"]["dataset"] == "OfficerChul/Android-Control-84k"
    assert payload["protocol"]["train_rows"] == 512
    assert payload["protocol"]["eval_rows"] == 256
    assert payload["comparison"]["warm_minus_random_after_pp"] > 19.0
    assert payload["comparison"]["exact_sequence_accuracy"] == {"warm": 0.0, "random": 0.0}
    assert payload["weight_adoption"]["warm_action_heads_frozen"] is True
    assert payload["weight_adoption"]["reuse_warm_backbone"] is True
    assert payload["source"]["screenshots"] == "omitted_from_projection"
