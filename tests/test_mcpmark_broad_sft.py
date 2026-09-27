import json
from pathlib import Path


def test_m120_broad_sft_receipt_binds_matched_transfer() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m120-mcpmark-broad-redacted-sft-transfer-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["dataset"]["source_rows"] == 10
    assert receipt["dataset"]["source_tool_calls"] == 131
    assert receipt["dataset"]["train_rows"] == 8
    assert receipt["dataset"]["eval_rows"] == 2
    assert receipt["dataset"]["train_services"] == ["filesystem", "notion", "github", "postgres"]
    assert receipt["dataset"]["eval_services"] == ["playwright"]
    assert receipt["normalized_data"]["redacted_tool_outputs"] is True
    assert receipt["normalized_data"]["redacted_assistant_text"] is True
    assert receipt["comparison"]["aggregate"]["warm_minus_random_after_pp"] > 39.0
    assert receipt["training"]["warm"]["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert receipt["training"]["random_backbone"]["after"]["eval"][
        "assistant_sequence_accuracy"
    ] == 0.0
    assert receipt["weight_analysis"]["warm_backbone_relative_delta_l2"] < 0.01
    assert receipt["weight_analysis"]["random_backbone_relative_delta_l2"] > 1.0
    assert "not an official MCPMark score" in receipt["claim_boundary"]
