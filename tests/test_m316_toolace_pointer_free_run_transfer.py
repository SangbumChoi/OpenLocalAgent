import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes


RECEIPT = Path("docs/paper/results/raw/m316-current-toolace-pointer-free-run-transfer-v1.json")


def test_m316_receipt_is_hash_bound_and_rejects_pointer_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert payload["dataset"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["pointer_training"]["train_locatable_spans"] == 182
    assert payload["pointer_training"]["eval_locatable_spans"] == 63
    assert payload["decision"]["pointer_span_improves"] is True
    assert payload["decision"]["adoption"] == "reject_pointer_promotion"


def test_m316_free_run_argument_and_episode_metrics_do_not_improve() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    before = payload["free_run"]["before"]["metrics"]
    after = payload["free_run"]["after"]["metrics"]
    assert after["argument_exact_rate"] == before["argument_exact_rate"]
    assert after["episode_exact_rate"] == before["episode_exact_rate"] == 0.0
