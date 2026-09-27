import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m263-workshop-gate-current-mobilegym-v1.json")


def test_m263_gate_accepts_current_mobilegym_but_remains_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["ready"] is False
    assert "native:mobilegym" in payload["passing_requirements"]
    assert "weights:transfer_and_no_transfer_ablation" in payload["passing_requirements"]
    blocked = {item["requirement"] for item in payload["blocked_requirements"]}
    assert blocked == {
        "native:androidworld",
        "native:mobile_safety_bench",
        "native:iosworld",
        "native:browsergym_miniwob",
        "native:osworld",
        "native:osworld_v2",
        "native:agentnet",
        "native:toolsandbox",
        "native:mcpmark",
        "native:enterpriseopsgym",
        "webgpu:native_capability_and_latency",
    }
