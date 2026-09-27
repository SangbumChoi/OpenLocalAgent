import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes


RECEIPT = Path("docs/paper/results/raw/m171-current-child-toolace-transfer-v1.json")


def test_m171_toolace_receipt_is_hash_bound_and_matched() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert payload["dataset"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["dataset"]["revision"] == "6bda777c88d21e5a204703c1ee45597a8fa4f734"
    assert payload["dataset"]["license"] == "Apache-2.0"
    assert payload["dataset"]["projection_manifest"]["accepted_rows"] == 8993
    assert payload["dataset"]["projection_manifest"]["rejected_rows"] == 2307
    assert payload["dataset"]["projection_manifest"]["split_audit"]["parent_record_overlap"] == 0
    assert payload["dataset"]["projection_manifest"]["split_audit"]["prompt_overlap"] == 0
    assert payload["bounded_arm"]["train_rows"] == 1024
    assert payload["bounded_arm"]["eval_rows"] == 256
    assert payload["compatibility"]["shared_tensor_count"] == 51
    assert payload["compatibility"]["tokenizer_sha256_equal"] is True
    assert payload["comparison"]["warm_start_better_after"] is True
    assert payload["comparison"]["warm_start"]["sequence_accuracy"] == 0.0
    assert payload["comparison"]["random_backbone"]["sequence_accuracy"] == 0.0
    assert payload["decision"] == "diagnostic_only"


def test_m171_toolace_receipt_keeps_native_and_public_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    boundary = payload["claim_boundary"]
    assert "official ToolACE" in boundary
    assert "native mobile" in boundary
    assert payload["publication"] == {
        "public_hub_url": None,
        "reason": "Hugging Face authentication is not configured.",
        "uploaded": False,
    }
