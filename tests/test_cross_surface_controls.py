"""Unit checks for matched cross-surface control comparison."""

from copy import deepcopy

import pytest

from scripts.compare_cross_surface_controls import _assert_matched, compare


def _receipt(mode: str, score: float) -> dict:
    source = {
        "label": "mobile",
        "rows": 2,
        "input": {"sha256": "a"},
        "public_reference": {"dataset": "d", "url": "https://example.test"},
    }
    metrics = {"rows": 2, "assistant_token_accuracy": score}
    return {
        "kind": "localagent_cross_surface_public_continuation_report",
        "parent": {"sha256": "parent"},
        "child": {"sha256": mode},
        "rows": {"train": 2, "eval": 2},
        "train_sources": [deepcopy(source)],
        "eval_sources": [deepcopy(source)],
        "hyperparameters": {
            "steps": 1,
            "batch_size": 1,
            "learning_rate": 1e-5,
            "max_seq_len": 32,
            "seed": 2027,
            "device": "cpu",
            "backbone_init": "parent" if mode == "warm" else "random",
            "random_backbone_seed": 2028,
        },
        "before": {"eval": metrics, "eval_by_source": {"mobile": metrics}},
        "after": {
            "eval": metrics,
            "eval_by_source": {"mobile": {"rows": 2, "assistant_token_accuracy": score}},
        },
        "weight_transfer": {"groups": {}},
    }


def test_compare_requires_matched_sources_and_reports_surface_delta() -> None:
    warm = _receipt("warm", 0.6)
    random = _receipt("random", 0.5)
    result = compare(warm, random)
    assert result["aggregate"]["warm_minus_random_after_pp"] == pytest.approx(10.0)
    assert result["surfaces"]["mobile"]["warm_start_better_after"] is True

    mismatched = deepcopy(random)
    mismatched["eval_sources"][0]["input"]["sha256"] = "different"
    with pytest.raises(ValueError, match="eval_sources mismatch"):
        _assert_matched(warm, mismatched)
