import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m338-realistic-agent-runtime-capability-audit-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m338_runtime_capability_receipt_is_self_consistent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["catalog"]["entries"] == 40
    assert payload["catalog"]["runnable_rows"] == 4
    assert payload["catalog"]["blocked_rows"] == 36
    assert payload["runnable_ids"] == [
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    ]


def test_m338_keeps_native_claims_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    probes = payload["dependency_probes"]
    assert probes["command:adb"] is False
    assert probes["command:docker"] is False
    assert probes["module:browsergym"] is False
    assert probes["module:mcpmark"] is False
    assert "claim an official score" in payload["claim_boundary"]
    for benchmark in ("androidworld", "osworld", "mcpmark", "toolsandbox"):
        assert payload["native_blockers"][benchmark]
