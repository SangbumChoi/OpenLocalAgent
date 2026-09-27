import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m260-hf-local-export-current-browser-context-v1.json")


def test_m260_current_browser_context_hf_bundle_is_complete_and_unpublished() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["parameters"] == 10524544
    assert payload["bundle"]["export_verified_locally"] is True
    assert payload["bundle"]["roundtrip_state_dict"] == {
        "missing_keys": [],
        "unexpected_keys": [],
        "tensor_count": 40,
    }
    assert payload["bundle"]["files"]["model.safetensors"]["bytes"] == 42101904
    assert payload["dispatch_metadata"] == {
        "tool_count": 63,
        "pointer_argument_count": 0,
        "heads_included": True,
        "surface_selectors_included": False,
        "provenance": "inferred_standard_tool_pool_from_51_class_tool_head",
    }
    assert payload["publication"] == {
        "published": False,
        "hub_url": None,
        "authenticated": False,
        "uploaded": False,
        "reason": "hf auth whoami reports no login; upload requires a user-provided Hugging Face token and repository",
    }
