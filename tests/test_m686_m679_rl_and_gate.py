import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _receipt(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return payload


def test_m686_rl_preflight_is_current_and_has_real_update_signal() -> None:
    payload = _receipt(ROOT / "docs/paper/results/raw/m686-m679-rl-preflight-v1.json")
    assert payload["kind"] == "localagent_one_update_training_preflight"
    assert payload["status"] == "passed"
    assert payload["metrics"]["lineage"]["parent_checkpoint_sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert payload["metrics"]["policy_transition"]["nonzero_learning_rate_executed"] is True
    assert payload["metrics"]["rl_accounting"]["realized_optimizer_updates"] == 2


def test_m686_gate_keeps_native_and_public_boundaries_explicit() -> None:
    payload = _receipt(ROOT / "docs/paper/results/raw/m686-workshop-gate-current-m679-v1.json")
    assert payload["ready"] is False
    assert "training:rl_preflight" in payload["passed_requirements"]
    assert "native:androidworld" in payload["blocked_requirements"]
    assert payload["blocked_requirements"]["artifacts:public_model_demo_manifest"] == "current_checkpoint_not_bound"
