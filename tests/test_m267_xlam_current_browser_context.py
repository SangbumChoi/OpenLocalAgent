import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m267-xlam-current-browser-context-v1.json")


def test_m267_current_checkpoint_xlam_derivative_receipt_is_hash_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["source"]["official_original_split_verified"] is False
    assert payload["source"]["rows_evaluated"] == 128
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["evaluator"]["first_call_only"] is True
    assert payload["modes"]["row_retriever"]["first_tool_exact_rate"] == 0.5078125
    assert payload["modes"]["row_retriever"]["schema_valid_rate"] == 1.0
    assert payload["modes"]["runtime_retriever_selector"]["first_tool_exact_rate"] == 0.1171875
    assert payload["modes"]["global_selector"]["first_tool_exact_rate"] == 0.0078125
    assert "not the gated Salesforce source" in payload["claim_boundary"]
