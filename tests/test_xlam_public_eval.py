import json
from pathlib import Path

from scripts.evaluate_xlam_public import ALL_MODES


def test_m115_receipt_distinguishes_public_derivative_from_gated_original() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m115-xlam-derived-function-calling-transfer-v1.json"
        ).read_text(encoding="utf-8")
    )
    source = receipt["source"]
    assert source["original_dataset"] == "Salesforce/xlam-function-calling-60k"
    assert source["original_access"] == "gated_and_not_authenticated_in_this_environment"
    assert source["derived_license"] == "Apache-2.0"
    assert source["official_original_split_verified"] is False
    assert source["training_used"] is False
    assert receipt["warm"]["row_retriever"]["first_tool_exact_rate"] == 0.5
    assert receipt["warm"]["row_retriever"]["schema_valid_rate"] == 1.0
    assert receipt["warm"]["row_retriever"]["first_arguments_exact_rate"] == 0.0
    assert receipt["warm"]["global_selector"]["first_tool_exact_rate"] > receipt["random_control"]["global_selector"]["first_tool_exact_rate"]


def test_xlam_evaluator_exposes_bounded_candidate_mode_set() -> None:
    assert ALL_MODES == (
        "row_retriever",
        "global_retriever",
        "runtime_retriever_selector",
        "global_selector",
    )
