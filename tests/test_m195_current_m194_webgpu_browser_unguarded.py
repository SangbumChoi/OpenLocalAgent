import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m195-current-m194-webgpu-browser-unguarded-v1.json"
)


def test_m195_binds_m194_export_and_separates_routing_from_grounding() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["parameters"] < 100_000_000
    assert payload["bundle"]["parity_gate_passed"] is True
    assert payload["browser"]["model_ready"] is True
    assert payload["browser"]["backend_badge"] == "WEBGPU"
    assert payload["browser"]["external_side_effects"] is False
    assert payload["quality"]["single_step"]["normalized_tool_family_exact_count"] == 5
    assert payload["quality"]["single_step"]["strict_action_exact_count"] == 1
    assert payload["quality"]["planner"]["strict_trajectory_exact_count"] == 0
    assert payload["decision"]["adopt_for_public_deployment"] is False
    assert "not an official BrowserGym" in payload["claim_boundary"]
