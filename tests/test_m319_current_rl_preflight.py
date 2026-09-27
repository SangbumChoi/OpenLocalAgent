import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


RECEIPT = Path("docs/paper/results/raw/m319-current-rl-preflight-v1.json")


def test_m319_current_checkpoint_rl_preflight_is_hash_bound_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "RLPreflightValidationError"
    assert payload["metrics"]["lineage"]["parent_checkpoint_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert "no reward diversity" in payload["error"]["message"]
    assert payload["metrics"]["heldout_eval"]["pre"]["tool_exact_match_accuracy"] == 0.0
