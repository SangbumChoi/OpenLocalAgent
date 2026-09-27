import pytest

from localagent.data.realistic_adapters import normalize_mobile_row
from localagent.eval.mobile import score_mobile_actions, score_mobile_row


def _row() -> dict:
    return {
        "record_id": "episode-1",
        "goal": "Open mail and tap search.",
        "steps": [
            {
                "instruction": "Open mail.",
                "accessibility_tree": "Mail",
                "action": {"action_type": "open_app", "app_name": "Mail"},
            },
            {
                "instruction": "Tap search.",
                "accessibility_tree": "Search",
                "action": {"action_type": "click", "x": 42, "y": 24},
            },
        ],
    }


def test_score_mobile_row_matches_normalized_calls() -> None:
    row = normalize_mobile_row(_row(), family="androidcontrol", source_revision="rev")
    score = score_mobile_row(
        row,
        [
            {"tool": "mobile_open_app", "args": {"app_name": "Mail"}},
            {"name": "mobile_click", "arguments": {"x": 42, "y": 24}},
        ],
    )
    assert score["tool_accuracy"] == 1.0
    assert score["action_exact_accuracy"] == 1.0
    assert score["trajectory_exact"] is True
    assert score["source_family"] == "androidcontrol"


def test_score_mobile_row_penalizes_missing_suffix() -> None:
    row = normalize_mobile_row(_row(), family="androidcontrol", source_revision="rev")
    score = score_mobile_row(row, [{"tool": "mobile_open_app", "args": {"app_name": "Mail"}}])
    assert score["tool_accuracy"] == 0.5
    assert score["action_exact_accuracy"] == 0.5
    assert score["trajectory_exact"] is False


def test_score_mobile_actions_reports_coordinate_and_grounded_scores() -> None:
    expected = [
        {"name": "mobile_click", "arguments": {"x": 10, "y": 20, "target_bbox": [8, 18, 12, 22]}}
    ]
    score = score_mobile_actions(expected, [{"tool": "mobile_click", "args": {"x": 11, "y": 21}}])
    assert score["action_exact_accuracy"] == 0.0
    assert score["coordinate_score_mean"] > 0.9
    assert score["grounded_score_mean"] == 1.0


def test_score_mobile_actions_rejects_empty_expected_stream() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        score_mobile_actions([], [])


def test_score_mobile_actions_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="finite"):
        score_mobile_actions(
            [{"name": "mobile_click", "arguments": {"x": 1, "y": 2}}],
            [{"tool": "mobile_click", "args": {"x": float("nan"), "y": 2}}],
        )
