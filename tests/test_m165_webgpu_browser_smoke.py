import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m165-webgpu-browser-smoke-v1.json")


def test_m165_browser_smoke_is_hash_bound_and_diagnostic_only() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["environment"]["backend"] == "webgpu"
    assert payload["environment"]["execution_provider_request"] == {
        "requested": "webgpu",
        "session_provider_count": 1,
        "single_provider_session_creation_succeeded": True,
        "whole_session_retry": False,
        "per_node_placement": "unknown",
        "per_node_fallback_status": "unknown",
        "note": "ORT Web does not expose per-node placement; this proves the requested session provider only.",
    }
    assert payload["bundle"]["parity_gate_passed"] is True
    assert payload["bundle"]["model_parameters"] == 10524544
    assert payload["run"]["records"] == 8
    assert payload["run"]["closed_loop_success_rate"] == 0.25
    assert payload["decision"] == "diagnostic_only"
    assert payload["publication"] == {
        "public_hub_url": None,
        "uploaded": False,
        "reason": "Hugging Face authentication is not configured.",
    }
