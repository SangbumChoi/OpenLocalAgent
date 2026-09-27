from __future__ import annotations

import json
from pathlib import Path

import pytest

from localagent.eval.enterpriseopsgym import load_tasks, summarize_scores


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"rows": [{"row": row} for row in rows]}, sort_keys=True),
        encoding="utf-8",
    )


def _row(task_id: str, tools: list[str]) -> dict:
    return {
        "task_id": task_id,
        "domain": "email",
        "system_prompt": "You are an email assistant.",
        "user_prompt": "Clean up the inbox.",
        "selected_tools": tools,
        "restricted_tools": [],
        "verifiers": "must never appear in output",
        "gym_servers_config": "secret server config",
    }


def test_enterprise_rows_drop_verifiers_and_require_matching_candidates(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.json"
    distractors = tmp_path / "plus15.json"
    _write_rows(oracle, [_row("task-1", ["list_messages", "list_messages"])])
    _write_rows(distractors, [_row("task-1", ["list_messages", "send_message", "send_message"])])

    tasks = load_tasks(oracle, distractors)
    assert len(tasks) == 1
    task = tasks[0]["task"]
    assert task.oracle_tools == ("list_messages",)
    assert task.candidate_tools == ("list_messages", "send_message")
    assert tasks[0]["verifiers_dropped"] is True
    assert tasks[0]["server_configuration_dropped"] is True
    assert "verifiers" not in task.__dict__


def test_enterprise_rows_reject_id_or_candidate_drift(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.json"
    distractors = tmp_path / "plus15.json"
    _write_rows(oracle, [_row("task-1", ["list_messages"])])
    _write_rows(distractors, [_row("task-2", ["list_messages"])])
    with pytest.raises(ValueError, match="task IDs must match"):
        load_tasks(oracle, distractors)

    _write_rows(distractors, [_row("task-1", ["send_message"])])
    with pytest.raises(ValueError, match="do not contain oracle tools"):
        load_tasks(oracle, distractors)


def test_enterprise_score_summary_is_deterministic_and_aggregate_only() -> None:
    scores = (
        {
            "task_id": "a",
            "domain": "email",
            "oracle_tool_count": 1,
            "candidate_tool_count": 3,
            "hit_at_1": True,
            "hit_at_3": True,
            "hit_at_5": True,
        },
        {
            "task_id": "b",
            "domain": "email",
            "oracle_tool_count": 2,
            "candidate_tool_count": 4,
            "hit_at_1": False,
            "hit_at_3": True,
            "hit_at_5": True,
        },
    )
    assert summarize_scores(scores) == {
        "records": 2,
        "hit_at_1": 0.5,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mean_oracle_tool_count": 1.5,
        "mean_candidate_tool_count": 3.5,
        "by_domain": {
            "email": {"records": 2, "hit_at_1": 0.5, "hit_at_3": 1.0, "hit_at_5": 1.0}
        },
    }


def test_m114_receipt_binds_warm_random_email_retrieval_and_claim_boundary() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m114-enterpriseopsgym-email-retrieval-m111-ablation-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["dataset"]["official_split_verified"] is False
    assert receipt["dataset"]["verifiers_dropped"] is True
    assert receipt["dataset"]["oracle_rows"] == 67
    assert receipt["warm"]["hit_at_1"] > receipt["random_control"]["hit_at_1"]
    assert receipt["warm"]["hit_at_3"] > receipt["random_control"]["hit_at_3"]
    assert receipt["warm"]["hit_at_5"] > receipt["random_control"]["hit_at_5"]
