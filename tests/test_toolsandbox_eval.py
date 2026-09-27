import json

import pytest

from localagent.eval.toolsandbox import aggregate_toolsandbox_results


def _row(name: str, similarity: float, categories: list[str] | None = None) -> dict:
    return {
        "name": name,
        "categories": categories or ["STATE_DEPENDENCY"],
        "traceback": None,
        "exception_type": None,
        "milestone_similarity": similarity,
        "minefield_similarity": 1.0,
        "similarity": similarity,
        "turn_count": 8,
        "milestone_mapping": {"0": [4, similarity]},
        "minefield_mapping": {},
    }


def test_toolsandbox_aggregate_verifies_complete_summary(tmp_path) -> None:
    path = tmp_path / "result_summary.json"
    path.write_text(
        json.dumps(
            [
                _row("send_message", 1.0),
                _row("insufficient_info", 0.5, ["INSUFFICIENT_INFORMATION"]),
            ]
        ),
        encoding="utf-8",
    )
    receipt = aggregate_toolsandbox_results(
        path,
        expected_scenarios=["send_message", "insufficient_info"],
    )
    assert receipt["status"] == "complete"
    assert receipt["completeness"]["verified"] is True
    assert receipt["overall"]["mean_similarity"] == 0.75
    assert receipt["overall"]["exact_similarity_rate"] == 0.5
    assert receipt["by_category"]["ALL_CATEGORIES"]["scenarios"] == 2


def test_toolsandbox_aggregate_fails_closed_on_missing_scenario(tmp_path) -> None:
    path = tmp_path / "result_summary.json"
    path.write_text(json.dumps([_row("send_message", 1.0)]), encoding="utf-8")
    receipt = aggregate_toolsandbox_results(path, expected_scenarios=["send_message", "not_seen"])
    assert receipt["status"] == "incomplete"
    assert receipt["completeness"]["missing_scenarios"] == ["not_seen"]


def test_toolsandbox_aggregate_matches_distraction_category_policy(tmp_path) -> None:
    path = tmp_path / "result_summary.json"
    path.write_text(
        json.dumps(
            [
                _row(
                    "scrambled",
                    0.4,
                    ["THREE_DISTRACTION_TOOLS", "TOOL_NAME_SCRAMBLED"],
                ),
                _row("plain", 1.0, ["THREE_DISTRACTION_TOOLS"]),
            ]
        ),
        encoding="utf-8",
    )
    receipt = aggregate_toolsandbox_results(path)
    assert receipt["by_category"]["THREE_DISTRACTION_TOOLS"]["scenarios"] == 1
    assert receipt["by_category"]["ALL_CATEGORIES"]["scenarios"] == 2


def test_toolsandbox_aggregate_rejects_malformed_mapping(tmp_path) -> None:
    path = tmp_path / "result_summary.json"
    row = _row("bad", 0.5)
    row["milestone_mapping"] = {"0": [1]}
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="must have two values"):
        aggregate_toolsandbox_results(path)
