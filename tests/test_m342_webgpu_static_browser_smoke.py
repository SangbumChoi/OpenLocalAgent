import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m342-webgpu-static-browser-smoke-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m342_receipt_and_static_banner_are_current() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["bundle"]["checkpoint_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["bundle"]["tool_count"] == 63
    assert payload["bundle"]["parity_gate_passed"] is True
    assert payload["browser"]["status"] == 200
    assert payload["browser"]["page_errors"] == 0
    assert payload["browser"]["visible_banner_bpe"] is True
    assert payload["browser"]["visible_banner_tool_count"] == 63


def test_m342_does_not_turn_wasm_fallback_into_webgpu_claim() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["browser"]["webgpu_adapter_available"] is False
    assert payload["browser"]["wasm_fallback_observed"] is True
    assert "not hardware WebGPU capability evidence" in payload["claim_boundary"]
