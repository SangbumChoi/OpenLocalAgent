import hashlib
import json
from pathlib import Path


ROOT = Path("docs/paper/results/raw")
CURRENT_SHA = "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"


def _load(name: str) -> dict:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    return payload


def test_m601_current_transfer_is_checkpoint_bound_and_warm_wins_all_sources() -> None:
    payload = _load("m601-m585-current-transfer-ablation-v1.json")
    assert payload["parent_checkpoint"]["sha256"] == CURRENT_SHA
    aggregate = payload["comparison"]["aggregate"]
    assert aggregate["warm_start_better_after"] is True
    assert aggregate["warm_minus_random_after_pp"] == 44.31119311193111
    assert all(row["warm_start_better_after"] for row in payload["comparison"]["surfaces"].values())
    assert payload["weight_transfer_analysis"]["warm"]["compatibility"]["shape_mismatches"] == {}


def test_m602_gate_accepts_current_transfer_but_remains_fail_closed() -> None:
    payload = _load("m602-workshop-gate-current-m585-transfer-v1.json")
    assert payload["ready"] is False
    assert "weights:transfer_and_no_transfer_ablation" in payload["passed_requirements"]
    assert "native:androidworld" in payload["blocked_requirements"]
    assert "artifacts:public_model_demo_manifest" in payload["blocked_requirements"]
