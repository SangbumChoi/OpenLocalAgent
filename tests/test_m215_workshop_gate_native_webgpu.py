import json
from pathlib import Path


def test_m215_gate_records_native_webgpu_pass_without_faking_publication() -> None:
    path = Path("docs/paper/results/raw/m215-workshop-gate-native-webgpu-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert len(payload["blocking_requirements"]) == 12
    checks = {item["requirement"]: item for item in payload["checks"]}
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["artifacts:public_model_demo_manifest"]["status"] == "blocked"
