import json
from pathlib import Path


_RAW = Path("docs/paper/results/raw")


def _load(name: str) -> dict:
    return json.loads((_RAW / name).read_text(encoding="utf-8"))


def test_m524_current_bundle_uses_hardware_webgpu_without_side_effects() -> None:
    receipt = _load("m524-webgpu-current-bundle-rerun-v1.json")
    assert receipt["backend"] == "webgpu"
    assert receipt["environment_executed"] is True
    assert receipt["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert receipt["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert receipt["capability"]["evaluated_cases"] == 3
    assert receipt["capability"]["exact_actions"] == 3
    assert receipt["capability"]["closed_loop_success"] == 0
    assert receipt["performance"]["tokens_per_second_p50"] > 600


def test_m525_gate_accepts_webgpu_but_remains_not_ready() -> None:
    receipt = _load("m525-workshop-gate-current-m524-v1.json")
    assert receipt["ready"] is False
    webgpu = next(
        check
        for check in receipt["checks"]
        if check["requirement"] == "webgpu:native_capability_and_latency"
    )
    assert webgpu["status"] == "pass"
    assert "native:androidworld" in {
        item["requirement"] for item in receipt["blocking_requirements"]
    }
