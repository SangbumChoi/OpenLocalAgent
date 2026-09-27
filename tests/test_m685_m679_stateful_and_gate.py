import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _self_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return payload


def test_m685_stateful_probe_binds_current_checkpoint_and_negative_boundary() -> None:
    payload = _self_hashed(ROOT / "docs/paper/results/raw/m685-m679-stateful-productivity-probe-v1.json")
    assert payload["kind"] == "localagent_m685_m679_stateful_productivity_probe"
    assert payload["parent_checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["configuration"]["native_runtime_executed"] is False
    assert payload["arms"]["pretrained_frozen_backbone"]["route_accuracy"] == 0.9375
    assert payload["arms"]["pretrained_frozen_backbone"]["task_complete_rate"] == 0.4
    assert payload["comparison"]["decision"].endswith("do_not_promote_native")


def test_m685_gate_is_fail_closed_and_lists_current_blockers() -> None:
    payload = _self_hashed(ROOT / "docs/paper/results/raw/m685-workshop-gate-current-m679-v1.json")
    assert payload["ready"] is False
    assert "native:androidworld" in payload["blocked_requirements"]
    assert payload["blocked_requirements"]["artifacts:public_model_demo_manifest"] == "current_checkpoint_not_bound"
    assert "native:mobilegym" in payload["passed_requirements"]
    assert "webgpu:native_capability_and_latency" in payload["passed_requirements"]
