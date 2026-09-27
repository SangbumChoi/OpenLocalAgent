import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m194-current-m180-dispatch-repair-v1.json"
)


def test_m194_records_a_non_promoted_warm_start_comparison() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["parent_checkpoint"]["parameters"] < 100_000_000
    assert payload["adapter"]["backbone_frozen"] is True
    assert payload["adapter"]["synthetic_adapter_only"] is True
    assert payload["evaluation"]["warm"]["route_accuracy"] > 0.99
    assert payload["evaluation"]["warm"]["selector_top1_accuracy"] < payload["evaluation"]["matched_random"]["selector_top1_accuracy"]
    assert payload["evaluation"]["warm"]["canonical_tool_exact"] < payload["evaluation"]["matched_random"]["canonical_tool_exact"]
    assert payload["decision"]["adopt_warm_child"] is False
    assert "not an official benchmark score" in payload["claim_boundary"]
