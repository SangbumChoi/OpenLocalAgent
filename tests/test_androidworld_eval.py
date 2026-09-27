import gzip
import pickle
from datetime import datetime
from pathlib import Path

import pytest

from localagent.eval.androidworld import aggregate_androidworld_run


def _write_group(root: Path, filename: str, episodes: list[dict]) -> None:
    with gzip.open(root / filename, "wb") as handle:
        pickle.dump(episodes, handle)


def _episode(task: str, instance: int, success: float, *, exception=None) -> dict:
    return {
        "task_template": task,
        "instance_id": instance,
        "is_successful": success,
        "run_time": 1.5,
        "episode_length": 3,
        "exception_info": exception,
    }


def test_androidworld_aggregate_verifies_complete_task_coverage(tmp_path) -> None:
    run_dir = tmp_path / "run_20260802T000000"
    run_dir.mkdir()
    _write_group(
        run_dir,
        "ClockStopWatchPaused_0.pkl.gz",
        [_episode("ClockStopWatchPaused", 0, 1)],
    )
    _write_group(
        run_dir,
        "ClockStopWatchPaused_1.pkl.gz",
        [_episode("ClockStopWatchPaused", 1, 0)],
    )
    receipt = aggregate_androidworld_run(
        run_dir,
        expected_tasks=["ClockStopWatchPaused"],
        n_task_combinations=2,
        source_revision="test-revision",
        agent_name="fixture",
    )
    assert receipt["status"] == "complete"
    assert receipt["completeness"]["verified"] is True
    assert receipt["overall"]["success_rate"] == 0.5
    assert receipt["by_task"]["ClockStopWatchPaused"]["episodes"] == 2
    assert receipt["loader"] == "builtin_only_pickle"


def test_androidworld_aggregate_fails_closed_on_missing_task_or_instance(tmp_path) -> None:
    run_dir = tmp_path / "run_20260802T000001"
    run_dir.mkdir()
    _write_group(
        run_dir,
        "ClockStopWatchPaused_0.pkl.gz",
        [_episode("ClockStopWatchPaused", 0, 1)],
    )
    receipt = aggregate_androidworld_run(
        run_dir,
        expected_tasks=["ClockStopWatchPaused", "ClockStopWatchRunning"],
        n_task_combinations=2,
    )
    assert receipt["status"] == "incomplete"
    assert receipt["completeness"]["missing_tasks"] == ["ClockStopWatchRunning"]
    assert receipt["completeness"]["count_mismatches"] == {"ClockStopWatchPaused": 1}


def test_androidworld_aggregate_counts_upstream_failed_result_as_failure(tmp_path) -> None:
    run_dir = tmp_path / "run_20260802T000002"
    run_dir.mkdir()
    _write_group(
        run_dir,
        "Task_0.pkl.gz",
        [_episode("Task", 0, float("nan"), exception="traceback")],
    )
    receipt = aggregate_androidworld_run(run_dir)
    assert receipt["overall"]["successes"] == 0
    assert receipt["overall"]["exception_episodes"] == 1
    assert receipt["completeness"]["verified"] is False


def test_androidworld_safe_loader_rejects_non_builtin_objects(tmp_path) -> None:
    run_dir = tmp_path / "run_20260802T000003"
    run_dir.mkdir()
    episode = _episode("Task", 0, 1)
    episode["finish_dtime"] = datetime.now()
    _write_group(run_dir, "Task_0.pkl.gz", [episode])
    with pytest.raises(ValueError, match="safe AndroidWorld pickle"):
        aggregate_androidworld_run(run_dir)
    receipt = aggregate_androidworld_run(run_dir, allow_unsafe_pickle=True)
    assert receipt["loader"] == "unsafe_pickle"


def test_androidworld_aggregate_rejects_symlink_result(tmp_path) -> None:
    run_dir = tmp_path / "run_20260802T000004"
    run_dir.mkdir()
    source = tmp_path / "source.pkl.gz"
    _write_group(tmp_path, source.name, [_episode("Task", 0, 1)])
    (run_dir / "Task_0.pkl.gz").symlink_to(source)
    with pytest.raises(ValueError, match="must not be a symlink"):
        aggregate_androidworld_run(run_dir)
