import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m348-toolsandbox-native-current-smoke-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m348_binds_current_checkpoint_and_pinned_native_verifier() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "toolsandbox"
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["environment"]["native_simulator_executed"] is True
    assert payload["environment"]["verifier_executed"] is True
    assert payload["environment"]["source_revision"] == (
        "165848b9a78cead7ca7fe7c89c688b58e6501219"
    )


def test_m348_keeps_official_split_and_productivity_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["environment"]["official_split_verified"] is False
    assert payload["environment"]["user_simulator_executed"] is False
    smoke = payload["protocols"]["single_step_smoke"]
    assert smoke["scenario_count"] == 3
    assert smoke["success_count"] == 2
    assert smoke["success_rate"] == 2 / 3
    interactive = payload["protocols"]["interactive"]
    assert interactive["scenario_count"] == 1
    assert interactive["success_count"] == 0
    assert "not an official ToolSandbox leaderboard score" in payload["claim_boundary"]
    assert "email, Notion, MCP" in payload["claim_boundary"]
