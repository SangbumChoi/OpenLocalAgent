"""Regression checks for frozen-backbone cross-surface dispatch-head probes."""

import json
from pathlib import Path


def _load(name: str) -> dict[str, object]:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    return json.loads((root / name).read_text())


def test_cross_surface_head_arms_are_matched_and_backbone_frozen() -> None:
    warm = _load("m103-cross-surface-warm-head-v1.json")
    random = _load("m104-cross-surface-random-head-v1.json")
    comparison = _load("m105-cross-surface-head-ablation-v1.json")
    assert warm["kind"] == "localagent_cross_surface_dispatch_head_report"
    assert random["kind"] == "localagent_cross_surface_dispatch_head_report"
    assert warm["rows"] == random["rows"] == {"train": 4629, "eval": 1041, "train_decisions": 4731}
    assert warm["hyperparameters"] == random["hyperparameters"]
    assert warm["hyperparameters"]["backbone_frozen"] is True
    assert random["hyperparameters"]["backbone_frozen"] is True
    assert comparison["kind"] == "localagent_cross_surface_dispatch_head_ablation_report"
    assert comparison["aggregate"]["warm_minus_random_selector_pp"] > 0.0
    assert comparison["surfaces"]["browser"]["warm_minus_random_after_selector_pp"] == 66.66666666666667
    assert comparison["surfaces"]["desktop"]["warm_minus_random_after_selector_pp"] == 0.0
    for receipt in (warm, random):
        groups = receipt["weight_transfer"]["groups"]
        for name in ("embedding", "attention_or_mixer", "ffn", "normalization"):
            assert groups[name]["relative_delta_l2"] == 0.0
    assert "not an official benchmark score" in comparison["claim_boundary"]
