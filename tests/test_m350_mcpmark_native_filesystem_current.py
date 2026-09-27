import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m350-mcpmark-native-filesystem-current-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m350_receipt_binds_current_checkpoint_and_native_mcp_fixture() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["dataset"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["model"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["environment"]["mcp_server_executed"] is True
    assert payload["environment_executed"] is True
    assert payload["task_count"] == 1
    assert payload["success_rate"] == 0.0
    assert payload["rollout"]["verifier_exit_code"] == 1


def test_m350_keeps_official_mcpmark_claim_closed_after_model_failure() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["environment"]["official_split_verified"] is False
    assert payload["environment"]["user_simulator_executed"] is False
    assert payload["rollout"]["model_completed_task"] is False
    assert "no official MCPMark split or leaderboard score is claimed" in payload["claim_boundary"]
