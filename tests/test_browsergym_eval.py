import json

import pytest

from localagent.eval.browsergym import aggregate_browsergym_episodes


def _episode(task_id: str, seed: int, reward: float, *, error=None) -> dict:
    return {
        "task_id": task_id,
        "seed": seed,
        "steps": [
            {
                "action": "click('12')",
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "error": error,
            },
            {
                "action": "send_msg_to_user('done')",
                "reward": reward,
                "terminated": True,
                "truncated": False,
                "error": None,
            },
        ],
        "final_reward": reward,
        "terminated": True,
        "truncated": False,
    }


def test_browsergym_aggregate_verifies_cases_and_rewards(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _episode("browsergym/miniwob.a", 11, 1.0),
                _episode("browsergym/miniwob.a", 17, 0.0, error="bad action"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = aggregate_browsergym_episodes(
        path,
        expected_cases=[
            "browsergym/miniwob.a@11",
            "browsergym/miniwob.a@17",
        ],
        source_revision="browsergym-revision",
        miniwob_revision="miniwob-revision",
    )
    assert receipt["status"] == "complete"
    assert receipt["overall"]["mean_final_reward"] == 0.5
    assert receipt["overall"]["success_rate"] == 0.5
    assert receipt["overall"]["action_errors"] == 1
    assert receipt["by_task"]["browsergym/miniwob.a"]["cases"] == 2


def test_browsergym_aggregate_fails_closed_on_missing_case(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(_episode("task", 1, 1.0)) + "\n", encoding="utf-8")
    receipt = aggregate_browsergym_episodes(path, expected_cases=["task@1", "task@2"])
    assert receipt["status"] == "incomplete"
    assert receipt["completeness"]["missing_cases"] == ["task@2"]


def test_browsergym_aggregate_rejects_nonterminal_episode(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    row = _episode("task", 1, 1.0)
    row["terminated"] = False
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must end with terminated"):
        aggregate_browsergym_episodes(path)


def test_browsergym_aggregate_rejects_final_reward_mismatch(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    row = _episode("task", 1, 1.0)
    row["final_reward"] = 0.5
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="final_reward must equal"):
        aggregate_browsergym_episodes(path)
