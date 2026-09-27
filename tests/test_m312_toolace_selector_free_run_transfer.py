import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes


RECEIPT = Path("docs/paper/results/raw/m312-current-toolace-selector-free-run-transfer-v1.json")


def test_m312_receipt_is_hash_bound_and_rejects_full_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert payload["dataset"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["selector_training"]["train_actions"] == 16
    assert payload["selector_training"]["eval_actions"] == 17
    assert payload["decision"]["selector_top1_improves"] is True
    assert payload["decision"]["free_run_tool_exact_improves"] is True
    assert payload["decision"]["adoption"] == "reject_full_policy_promotion"


def test_m312_free_run_keeps_argument_and_episode_exactness_zero() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    before = payload["free_run"]["before"]["metrics"]
    after = payload["free_run"]["after"]["metrics"]
    assert before["argument_exact_rate"] == 0.0
    assert after["argument_exact_rate"] == 0.0
    assert before["episode_exact_rate"] == 0.0
    assert after["episode_exact_rate"] == 0.0
