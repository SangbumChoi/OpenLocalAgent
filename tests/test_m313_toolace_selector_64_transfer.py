import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes


RECEIPT = Path("docs/paper/results/raw/m313-current-toolace-selector-64-transfer-v1.json")


def test_m313_receipt_is_hash_bound_and_source_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert payload["dataset"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["dataset"]["train"]["bytes"] == 1346014
    assert payload["dataset"]["eval"]["bytes"] == 344723
    assert payload["selector_training"]["train_actions"] == 447
    assert payload["selector_training"]["eval_actions"] == 113


def test_m313_transfer_improves_but_remains_below_full_policy_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    before = payload["free_run"]["before"]["metrics"]
    after = payload["free_run"]["after"]["metrics"]
    assert payload["decision"]["adoption"] == "reject_full_policy_promotion"
    assert after["tool_exact_rate"] > before["tool_exact_rate"]
    assert after["episode_exact_rate"] > before["episode_exact_rate"]
    assert after["episode_exact_rate"] < 0.5
