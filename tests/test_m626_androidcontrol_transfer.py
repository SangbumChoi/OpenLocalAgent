import hashlib
import json
from pathlib import Path


def test_m626_androidcontrol_receipt_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m626-androidcontrol-warm-random-transfer-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["protocol"]["train_rows"] == 512
    assert payload["protocol"]["eval_rows"] == 256
    assert payload["protocol"]["visual_input_omitted"] is True
    assert payload["decision"]["native_mobile_promotion"] is False
    assert payload["comparison"]["aggregate"]["warm_start_better_after"] is True
