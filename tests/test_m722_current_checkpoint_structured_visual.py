import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m722-current-checkpoint-structured-visual-v1.json"


def test_m722_binds_current_checkpoint_and_rejects_visual_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m722_current_checkpoint_structured_visual"
    assert payload["checkpoint"]["sha256"] == "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    assert payload["export"]["cpu_onnx_parity"]["passed"] is True
    assert payload["weight_transfer"]["warm_minus_random_action_accuracy"] == 0.0
    assert payload["weight_transfer"]["warm_minus_random_coordinate_mae"] > 0
    assert payload["weight_transfer"]["decision"].startswith("do_not_promote")
