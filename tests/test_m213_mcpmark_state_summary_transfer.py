import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m213_receipt_hash_and_transfer_boundary() -> None:
    path = Path("docs/paper/results/raw/m213-mcpmark-state-summary-transfer-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.pop("receipt_self_sha256")
    assert claimed == _canonical_sha256(receipt)
    assert receipt["dataset"]["train"]["rows"] == 8
    assert receipt["dataset"]["eval"]["rows"] == 2
    assert receipt["dataset"]["state_policy"] == "status_shape_digest_only; tool result text, URLs, document content, identifiers, and assistant free text are not retained"
    assert receipt["training"]["warm"]["eval"]["after_sequence_accuracy"] == 0.0
    assert receipt["selector_probe"]["warm_eval"]["top1"] == 0.0
    assert receipt["decision"]["export_child"] is False
