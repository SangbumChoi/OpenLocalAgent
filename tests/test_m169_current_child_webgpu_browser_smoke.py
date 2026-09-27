import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m169-current-child-webgpu-browser-smoke-v1.json")


def test_m169_current_child_browser_smoke_is_hash_bound_and_diagnostic_only() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["environment"]["backend"] == "webgpu"
    assert payload["environment"]["requested_backend"] == "webgpu"
    assert payload["environment"]["execution_provider_request"] == {
        "requested": "webgpu",
        "session_provider_count": 1,
        "single_provider_session_creation_succeeded": True,
        "whole_session_retry": False,
        "per_node_placement": "unknown",
        "per_node_fallback_status": "unknown",
        "note": (
            "ORT Web does not expose per-node placement; this proves the requested session "
            "provider and does not claim every node executed on the GPU."
        ),
    }
    assert payload["bundle"]["parity_gate_passed"] is True
    assert payload["bundle"]["deployment_verification"]["blockers"] == []
    assert payload["bundle"]["model_parameters"] == 10524544
    assert payload["run"]["records"] == 8
    assert payload["run"]["schema_valid_rate"] == 1.0
    assert payload["run"]["closed_loop_success_rate"] == 0.25
    assert payload["run"]["exact_action_rate"] == 0.25
    assert payload["decision"] == "diagnostic_only"
    assert payload["publication"] == {
        "public_hub_url": None,
        "uploaded": False,
        "reason": "Hugging Face authentication is not configured.",
    }


def test_m169_browser_scope_excludes_native_and_external_claims() -> None:
    scope = json.loads(RECEIPT.read_text(encoding="utf-8"))["environment"]["benchmark_scope"]
    assert scope["input_modality"] == "text-only prompt"
    assert scope["visual_grounding"] is False
    assert scope["multi_step_planning"] is False
    assert scope["browser_wide_control"] is False
    assert scope["external_navigation"] is False
