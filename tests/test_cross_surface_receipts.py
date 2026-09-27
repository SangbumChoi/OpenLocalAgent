"""Regression checks for the source-bound cross-surface continuation receipt."""

import json
from pathlib import Path


def test_cross_surface_receipt_binds_public_sources_and_mixed_transfer() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    receipt = json.loads((root / "m100-cross-surface-public-continuation-v1.json").read_text())
    assert receipt["kind"] == "localagent_cross_surface_public_continuation_report"
    assert receipt["rows"] == {"train": 4629, "eval": 1041}
    assert [source["label"] for source in receipt["train_sources"]] == [
        "mobile",
        "desktop",
        "browser",
    ]
    assert [source["label"] for source in receipt["eval_sources"]] == [
        "mobile",
        "desktop",
        "browser",
    ]
    references = {
        source["label"]: source["public_reference"] for source in receipt["train_sources"]
    }
    assert references["mobile"]["dataset"] == "OfficerChul/Android-Control-84k"
    assert references["desktop"]["dataset"] == "xlangai/AgentNet"
    assert references["browser"]["dataset"] == "osunlp/Mind2Web"
    assert receipt["before"]["eval"]["assistant_token_accuracy"] < receipt["after"]["eval"]["assistant_token_accuracy"]
    assert receipt["before"]["eval_by_source"]["mobile"]["assistant_token_accuracy"] < receipt["after"]["eval_by_source"]["mobile"]["assistant_token_accuracy"]
    assert receipt["before"]["eval_by_source"]["desktop"]["assistant_token_accuracy"] > receipt["after"]["eval_by_source"]["desktop"]["assistant_token_accuracy"]
    groups = receipt["weight_transfer"]["groups"]
    assert groups["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.004 < groups["embedding"]["relative_delta_l2"] < 0.006
    assert "no official benchmark score" in receipt["claim_boundary"]


def test_cross_surface_matched_random_control_binds_all_surfaces() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    random_receipt = json.loads(
        (root / "m101-cross-surface-random-continuation-v1.json").read_text()
    )
    comparison = json.loads(
        (root / "m102-cross-surface-transfer-ablation-v1.json").read_text()
    )
    assert random_receipt["hyperparameters"]["backbone_init"] == "random"
    assert random_receipt["hyperparameters"]["random_backbone_seed"] == 2028
    assert random_receipt["rows"] == {"train": 4629, "eval": 1041}
    assert comparison["arm_contract"] == {
        "warm_backbone_init": "parent",
        "random_backbone_init": "random",
        "random_backbone_seed": 2028,
    }
    assert comparison["decision"] == "warm_start_dominates_matched_random_on_all_surfaces"
    assert comparison["aggregate"]["warm_minus_random_after_pp"] == 30.763773102983432
    assert {
        label: round(value["warm_minus_random_after_pp"], 2)
        for label, value in comparison["surfaces"].items()
    } == {"mobile": 27.09, "desktop": 47.11, "browser": 60.0}
    assert all(
        value["warm_start_better_after"] for value in comparison["surfaces"].values()
    )
