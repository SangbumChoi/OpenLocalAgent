import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m266-workshop-gate-current-mobile-browser-webgpu-v1.json"
)


def test_m266_gate_records_native_webgpu_and_remains_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["ready"] is False
    assert set(payload["passing_requirements"]) == {
        "catalog:realistic_family_coverage",
        "catalog:no_pending_train_adapter",
        "native:mobilegym",
        "native:browsergym_miniwob",
        "webgpu:native_capability_and_latency",
        "weights:transfer_and_no_transfer_ablation",
        "artifacts:public_model_demo_manifest",
    }
    blocked = {item["requirement"] for item in payload["blocked_requirements"]}
    assert blocked == {
        "native:androidworld",
        "native:mobile_safety_bench",
        "native:iosworld",
        "native:osworld",
        "native:osworld_v2",
        "native:agentnet",
        "native:toolsandbox",
        "native:mcpmark",
        "native:enterpriseopsgym",
    }
    webgpu = payload["webgpu_summary"]
    assert webgpu["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert webgpu["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert webgpu["evaluated_cases"] == 3
    assert webgpu["exact_actions"] == 3
    assert webgpu["measured_repetitions_per_case"] == 30
    assert webgpu["external_side_effects_executed"] is False
    assert webgpu["closed_loop_success"] == 0
