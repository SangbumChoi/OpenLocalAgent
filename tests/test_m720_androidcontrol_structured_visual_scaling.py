import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m720-androidcontrol-structured-visual-scaling-v1.json"


def test_m720_scaling_receipt_has_disjoint_holdout_and_rejects_transfer_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m720_androidcontrol_structured_visual_scaling"
    source = payload["source"]
    assert source["complete_episodes"] == source["train_episodes"] + source["heldout_episodes"]
    assert source["train_rows"] > source["heldout_rows"] > 0
    assert payload["weight_transfer"]["action_accuracy_delta_warm_minus_random"] == 0.0
    assert payload["weight_transfer"]["coordinate_mae_delta_warm_minus_random"] > 0
    assert payload["claim_boundary"].startswith("Official AndroidControl")
