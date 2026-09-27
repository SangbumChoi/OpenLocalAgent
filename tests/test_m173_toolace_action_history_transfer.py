import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes


RECEIPT = Path("docs/paper/results/raw/m173-current-child-toolace-action-history-transfer-v1.json")


def test_m173_toolace_action_history_receipt_is_hash_bound_and_matched() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert payload["kind"] == "localagent_current_child_toolace_action_history_transfer_receipt"
    assert payload["measurement"] == "m173_current_child_toolace_action_history_transfer"
    projection = payload["dataset"]["projection_manifest"]
    assert projection["adapter_version"] == "toolace-action-history-canonical-actions-v1"
    assert projection["projection_mode"] == "action_history"
    assert projection["accepted_rows"] == 8992
    assert projection["rejected_rows"] == 2308
    assert projection["projection_stats"] == {
        "assistant_action_turns": 9679,
        "assistant_text_turns_omitted": 1806,
        "messages": 21164,
        "tool_response_messages": 1357,
    }
    assert projection["split_audit"]["parent_record_overlap"] == 0
    assert projection["split_audit"]["prompt_overlap"] == 0
    assert payload["bounded_arm"]["train_rows"] == 256
    assert payload["bounded_arm"]["eval_rows"] == 64
    assert payload["compatibility"]["shared_tensor_count"] == 51
    assert payload["compatibility"]["tokenizer_sha256_equal"] is True
    assert payload["comparison"]["warm_start_better_after"] is True
    assert payload["comparison"]["warm_start"]["after_token_accuracy"] > 0.49
    assert payload["comparison"]["warm_start"]["sequence_accuracy"] == 0.0
    assert payload["comparison"]["random_backbone"]["sequence_accuracy"] == 0.0
    assert payload["decision"] == "diagnostic_only"


def test_m173_toolace_action_history_keeps_execution_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    boundary = payload["claim_boundary"]
    assert "action-history projection" in boundary
    assert "multi-turn execution result" in boundary
    assert "native mobile" in boundary
    assert payload["publication"] == {
        "public_hub_url": None,
        "reason": "Hugging Face authentication is not configured.",
        "uploaded": False,
    }
