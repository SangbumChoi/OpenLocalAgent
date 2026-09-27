import json
from pathlib import Path


def test_m119_dynamic_selector_receipt_is_a_frozen_transfer_diagnostic() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m119-mcpmark-dynamic-selector-transfer-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["source"]["train_rows"] == 2
    assert receipt["source"]["eval_rows"] == 1
    assert receipt["source"]["train_decisions"] == 51
    assert receipt["source"]["eval_decisions"] == 20
    assert receipt["source"]["license"] == "MIT"
    assert receipt["source"]["url"].endswith("Jakumetsu/mcpmark-trajectory-log")
    assert receipt["source"]["tool_outputs_redacted"] is True
    assert receipt["hyperparameters"]["frozen_backbone"] is True
    assert receipt["model_contract"]["parameter_count"] == 10524544
    warm_eval = receipt["arms"]["warm"]["eval"]
    random_eval = receipt["arms"]["random_backbone"]["eval"]
    assert warm_eval["tool_decisions"] == random_eval["tool_decisions"] == 19
    assert warm_eval["top10"]["accuracy"] > random_eval["top10"]["accuracy"]
    assert warm_eval["top1"]["accuracy"] == random_eval["top1"]["accuracy"] == 0.0
    assert receipt["weight_analysis"]["warm_random_backbone_state_exact"] is False
    assert "not an official MCPMark result" in receipt["claim_boundary"]
