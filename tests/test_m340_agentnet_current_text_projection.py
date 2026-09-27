import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m340-agentnet-current-text-projection-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m340_current_checkpoint_and_projection_are_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["projection"]["action_rows"] == 133
    assert payload["projection"]["eval_parent_records"] == 8
    assert payload["projection"]["images_consumed"] is False


def test_m340_keeps_desktop_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["decision"] == "retain_diagnostic_only"
    assert payload["metrics"]["exact_trajectory_rate"] == 0.0
    assert payload["metrics"]["first_action_type_rate"] == 0.0
    assert "not an official AgentNetBench leaderboard result" in payload["claim_boundary"]
    assert "native desktop success" in payload["claim_boundary"]
