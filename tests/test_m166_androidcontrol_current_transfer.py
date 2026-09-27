import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m166-androidcontrol-current-transfer-v1.json")


def test_m166_androidcontrol_transfer_is_hash_bound_and_non_visual() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["source"]["train"]["rows"] == 4096
    assert payload["source"]["eval"]["rows"] == 904
    assert payload["source"]["train"]["visual_input_omitted"] is True
    assert payload["source"]["eval"]["visual_input_omitted"] is True
    assert payload["training"]["after_eval"]["assistant_token_accuracy"] > payload[
        "training"
    ]["before_eval"]["assistant_token_accuracy"]
    assert payload["training"]["sequence_accuracy"] == {"before": 0.0, "after": 0.0}
    assert payload["weight_transfer"]["compatibility"] == {
        "config_mismatches": {},
        "shape_mismatches": {},
        "shared_tensor_count": 51,
        "tokenizer_sha256_equal": True,
    }
    assert payload["decision"] == "diagnostic_only"
