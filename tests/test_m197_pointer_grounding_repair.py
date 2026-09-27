import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m197-pointer-grounding-repair-v1.json")


def test_m197_records_matched_pointer_transfer_without_adoption() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["parent"]["parameters"] < 100_000_000
    assert payload["training"]["backbone_frozen"] is True
    assert payload["warm"]["eval_correct"] < payload["matched_random"]["eval_correct"]
    assert payload["warm"]["shared_pointer_relative_l2"] < payload["matched_random"]["shared_pointer_relative_l2"]
    assert payload["decision"]["adopt_warm"] is False
    assert "not an official Mind2Web test score" in payload["claim_boundary"]
