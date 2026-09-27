import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads(Path("docs/paper/results/raw", name).read_text(encoding="utf-8"))


def test_m151_coordinate_canary_is_explicitly_non_official_and_grounded() -> None:
    receipt = _load("m151-browsergym-native-coordinate-canary-v1.json")
    assert receipt["benchmark_id"] == "browsergym_miniwob"
    assert receipt["environment_executed"] is True
    assert receipt["coordinate_fallback"] is True
    assert receipt["official_split_verified"] is False
    assert receipt["task_count"] == 10
    assert receipt["success_rate"] == 0.4
    assert receipt["checkpoint"]["sha256"] == (
        "dc360a87a6af3b02e7a38be7a46ebe1bad98a0a27a31b77e15e53f7d2d88a183"
    )
    successes = [case for case in receipt["cases"] if case["success"]]
    assert len(successes) == 4
    assert {case["task"] for case in successes} == {"miniwob.ascending-numbers"}
    assert all(any(step["action"].startswith("mouse_click(") for step in case["steps"]) for case in successes)


def test_m152_full_coordinate_diagnostic_does_not_replace_official_baseline() -> None:
    receipt = _load("m152-browsergym-native-coordinate-full-m148-v1.json")
    assert receipt["benchmark_id"] == "browsergym_miniwob"
    assert receipt["environment_executed"] is True
    assert receipt["coordinate_fallback"] is True
    assert receipt["official_split_verified"] is False
    assert receipt["task_count"] == 240
    assert receipt["success_rate"] == 4 / 240
    assert receipt["checkpoint"]["sha256"] == (
        "dc360a87a6af3b02e7a38be7a46ebe1bad98a0a27a31b77e15e53f7d2d88a183"
    )
    successes = [case for case in receipt["cases"] if case["success"]]
    assert len(successes) == 4
    assert {case["task"] for case in successes} == {"miniwob.ascending-numbers"}
    steps = [step for case in receipt["cases"] for step in case["steps"]]
    assert sum(bool(step["grounded"]) for step in steps) == 370
    assert "non-official" in receipt["claim_boundary"]


def test_m521_current_checkpoint_coordinate_semantic_canary_is_non_official() -> None:
    receipt = _load("m521-browsergym-current-coordinate-semantic-canary-v1.json")
    assert receipt["benchmark_id"] == "browsergym_miniwob"
    assert receipt["environment_executed"] is True
    assert receipt["coordinate_fallback"] is True
    assert receipt["semantic_fallback"] is True
    assert receipt["official_split_verified"] is False
    assert receipt["task_count"] == 1
    assert receipt["success_rate"] == 1.0
    assert receipt["checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    case = receipt["cases"][0]
    assert case["success"] is True
    assert all(step["grounded"] for step in case["steps"])
    assert "non-official" in receipt["claim_boundary"]
