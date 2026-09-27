import hashlib
import json
from pathlib import Path

from localagent.data.conversation_artifact import canonical_json_bytes


RECEIPT = Path("docs/paper/results/raw/m174-current-child-toolace-action-history-free-run-v1.json")


def test_m174_toolace_free_run_receipt_is_hash_bound_and_bounded() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert payload["kind"] == "localagent_toolace_action_history_free_run_probe"
    assert payload["source"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["source"]["revision"] == "6bda777c88d21e5a204703c1ee45597a8fa4f734"
    assert payload["source"]["training_used"] is False
    assert payload["rows_requested"] == 16
    assert payload["rows_evaluated"] == 16
    assert payload["metrics"] == {
        "argument_exact_rate": 0.03333333333333333,
        "by_turn": {
            "1": {"step_exact_rate": 0.0625, "steps": 16},
            "2": {"step_exact_rate": 0.0, "steps": 12},
            "3": {"step_exact_rate": 0.0, "steps": 2},
        },
        "episode_exact_rate": 0.0,
        "episodes": 16,
        "schema_valid_rate": 0.6,
        "step_exact_rate": 0.03333333333333333,
        "steps": 30,
        "tool_exact_rate": 0.1,
    }
    assert all("tool_catalog" not in json.dumps(row) for row in payload["predictions"])


def test_m174_toolace_free_run_claim_boundary_closes_side_effects() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert "no tool dispatch" in payload["claim_boundary"]
    assert "official ToolACE" in payload["claim_boundary"]
    assert "external side effects" in payload["claim_boundary"]
