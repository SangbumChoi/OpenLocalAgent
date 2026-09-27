from scripts.normalize_appworld_trajectories import _actions, _safe_value
from scripts.evaluate_appworld_checkpoint import _parse_appworld_printed_result


def test_short_trajectory_can_retain_completion_without_bootstrap_calls() -> None:
    trace = [
        {"app": "supervisor", "api": "show_profile", "arguments": {}},
        {"app": "spotify", "api": "search_artists", "arguments": {"query": "Lily"}},
        {"app": "supervisor", "api": "complete_task", "arguments": {"status": "success"}},
    ]
    assert [f"{item['app']}.{item['api']}" for item in _actions(trace, 4)] == [
        "spotify.search_artists"
    ]
    assert [f"{item['app']}.{item['api']}" for item in _actions(trace, 4, include_completion=True)] == [
        "spotify.search_artists",
        "supervisor.complete_task",
    ]


def test_compact_state_keeps_answer_relevant_follower_count() -> None:
    assert _safe_value({"artist_id": 24, "follower_count": 20, "created_at": "omit"}) == {
        "artist_id": 24,
        "follower_count": 20,
    }


def test_persisted_appworld_result_parser_accepts_literal_output_only() -> None:
    assert _parse_appworld_printed_result("{'message': 'ok'}\n") == {"message": "ok"}
