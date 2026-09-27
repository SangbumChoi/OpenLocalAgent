import json
import hashlib
from pathlib import Path


RECEIPT = Path(__file__).parents[1] / "docs/paper/results/raw/m26-hf-local-export-v2.json"


def test_hf_local_export_receipt_is_not_a_publication_claim() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_hf_local_export_receipt"
    assert payload["publication"] == {
        "published": False,
        "hub_url": None,
        "reason": "HF authentication was not configured; this receipt binds a local bundle only.",
    }
    assert payload["bundle"]["parameters"] == 10524544
    assert payload["bundle"]["tokenizer"]["kind"] == "bpe"
    assert set(payload["bundle"]["head_keys"]) == {
        "dense_selector",
        "ptr_head",
        "route_head",
        "selector_proj",
        "tool_head",
    }


def test_m131_mobile_export_binds_dispatch_metadata_without_claiming_upload() -> None:
    payload = json.loads(
        Path("docs/paper/results/raw/m131-hf-local-export-m129-mobile-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["kind"] == "localagent_hf_local_export_receipt"
    assert payload["parameter_count"] == 10524544
    assert payload["dispatch_tool_count"] == 63
    assert payload["tokenizer_sha256"] == "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c"
    assert payload["export_verified_locally"] is True
    assert payload["hub"] == {
        "authenticated": False,
        "uploaded": False,
        "reason": "hf auth whoami reports no login; upload requires a user-provided Hugging Face token and repository",
    }
    assert payload["bundle"]["agent_heads.bin"]["bytes"] == 10137222


def test_m164_current_child_hf_export_is_complete_but_unpublished() -> None:
    path = Path("docs/paper/results/raw/m164-hf-local-export-current-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["parameters"] == 10524544
    assert payload["bundle"]["export_verified_locally"] is True
    assert payload["dispatch_metadata"] == {
        "tool_count": 63,
        "pointer_argument_count": 23,
        "provenance": "inferred_standard_tool_pool_from_51_class_tool_head",
        "model_card_reports_tools": 63,
        "heads_included": True,
    }
    assert payload["publication"] == {
        "published": False,
        "hub_url": None,
        "authenticated": False,
        "uploaded": False,
        "reason": "hf auth whoami reports no login; upload requires a user-provided Hugging Face token and repository",
    }


def test_m330_current_checkpoint_export_binds_local_hf_bundle_without_claiming_upload() -> None:
    path = Path("docs/paper/results/raw/m330-hf-local-export-current-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["checkpoint"]["parameters"] == 10524544
    assert payload["bundle"]["export_verified_locally"] is True
    assert payload["bundle"]["files"]["config.json"]["sha256"] == (
        "29ac9528d9af3b329fe4628966cf4fa0347f54060cceb18b74664fa6c8e0ab56"
    )
    assert payload["dispatch_metadata"] == {
        "heads_included": True,
        "pointer_argument_count": 23,
        "provenance": "inferred_standard_tool_pool_from_51_class_tool_head",
        "tool_count": 63,
    }
    assert payload["publication"]["published"] is False
    assert payload["publication"]["uploaded"] is False
