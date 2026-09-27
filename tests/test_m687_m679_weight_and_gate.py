import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _receipt(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert expected == actual
    return payload


def test_m687_weight_transfer_binds_current_parent_and_both_arms() -> None:
    payload = _receipt(ROOT / "docs/paper/results/raw/m687-m679-current-weight-transfer-v1.json")
    assert payload["parent_checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
    assert set(payload["weight_transfer_analysis"]) == {"warm", "random"}
    assert payload["weight_transfer_analysis"]["warm"]["compatibility"]["tokenizer_sha256_equal"] is True
    assert payload["weight_transfer_analysis"]["warm"]["groups"]["embedding"]["relative_delta_l2"] == 0.0
    assert payload["weight_transfer_analysis"]["random"]["groups"]["embedding"]["relative_delta_l2"] > 1.0
    assert payload["comparison"]["arm_contract"]["warm_backbone_init"] == "parent"


def test_m687_gate_only_blocks_native_suites_and_public_artifact() -> None:
    payload = _receipt(ROOT / "docs/paper/results/raw/m687-workshop-gate-current-m679-v1.json")
    assert payload["ready"] is False
    assert "weights:transfer_and_no_transfer_ablation" in payload["passed_requirements"]
    assert "training:rl_preflight" in payload["passed_requirements"]
    assert payload["blocked_requirements"]["artifacts:public_model_demo_manifest"] == "current_checkpoint_not_bound"
    assert len(payload["blocked_requirements"]) == 10
