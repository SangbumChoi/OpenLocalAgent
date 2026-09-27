import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m714-androidcontrol-structured-visual-pilot-v1.json")


def test_m714_binds_structured_visual_action_metrics() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m714_androidcontrol_structured_visual_pilot"
    assert payload["model"]["parameters"] < 100_000_000
    assert payload["split"]["parent_disjoint"] is True
    assert payload["split"]["train_samples"] > 0
    assert payload["split"]["eval_samples"] > 0
    assert 0 <= payload["warm"]["after"]["action_accuracy"] <= 1
    assert 0 <= payload["random"]["after"]["action_accuracy"] <= 1


def test_m714_does_not_promote_without_native_validation() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["weight_analysis"]["adoption_decision"].startswith("do_not_promote")
    assert "native verifier" in payload["claim_boundary"]
