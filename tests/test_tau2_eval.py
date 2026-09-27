import json

import pytest

from localagent.eval.tau2 import aggregate_tau2_results


def _simulation(task_id: str, trial: int, reward: float, *, termination="agent_stop") -> dict:
    return {
        "id": f"{task_id}-{trial}",
        "task_id": task_id,
        "trial": trial,
        "termination_reason": termination,
        "duration": 1.25,
        "agent_cost": 0.01,
        "reward_info": {"reward": reward},
        "messages": [{"role": "user", "content": "hello"}],
    }


def _metadata(simulations: list[dict], *, trials=2) -> dict:
    return {
        "info": {
            "num_trials": trials,
            "environment_info": {"domain_name": "retail"},
        },
        "tasks": [{"id": "a"}, {"id": "b"}],
        "simulations": simulations,
    }


def test_tau2_monolithic_result_reports_upstream_pass_hat_k(tmp_path) -> None:
    path = tmp_path / "results.json"
    simulations = [
        _simulation("a", 1, 1.0),
        _simulation("a", 2, 0.0),
        _simulation("b", 1, 1.0),
        _simulation("b", 2, 0.0),
    ]
    path.write_text(json.dumps(_metadata(simulations)), encoding="utf-8")
    receipt = aggregate_tau2_results(
        path,
        expected_cases=["retail/a@1", "retail/a@2", "retail/b@1", "retail/b@2"],
        expected_trials=2,
        source_revision="tau2-revision",
    )
    assert receipt["status"] == "complete"
    assert receipt["overall"]["average_reward"] == 0.5
    assert receipt["overall"]["pass_hat_k"] == {"1": 0.5, "2": 0.0}
    assert receipt["by_task"]["retail/a"]["pass_hat_k"] == {"1": 0.5, "2": 0.0}


def test_tau2_directory_result_reads_individual_simulations(tmp_path) -> None:
    root = tmp_path / "run"
    simulations = [_simulation("a", 1, 1.0), _simulation("a", 2, 1.0)]
    (root / "simulations").mkdir(parents=True)
    metadata = _metadata([], trials=2)
    metadata.pop("simulations")
    (root / "results.json").write_text(json.dumps(metadata), encoding="utf-8")
    for simulation in simulations:
        (root / "simulations" / f"{simulation['id']}.json").write_text(
            json.dumps(simulation), encoding="utf-8"
        )
    receipt = aggregate_tau2_results(
        root,
        expected_cases=["retail/a@1", "retail/a@2"],
    )
    assert receipt["status"] == "complete"
    assert receipt["overall"]["pass_hat_k"] == {"1": 1.0, "2": 1.0}
    assert len(receipt["source_files"]) == 3


def test_tau2_result_marks_missing_case_incomplete(tmp_path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(_metadata([_simulation("a", 1, 1.0)])),
        encoding="utf-8",
    )
    receipt = aggregate_tau2_results(
        path,
        expected_cases=["retail/a@1", "retail/a@2"],
        expected_trials=2,
    )
    assert receipt["status"] == "incomplete"
    assert receipt["completeness"]["missing_cases"] == ["retail/a@2"]
    assert receipt["completeness"]["count_mismatches"] == {"retail/a": 1}


def test_tau2_result_excludes_infrastructure_errors_and_fails_closed(tmp_path) -> None:
    path = tmp_path / "results.json"
    simulations = [
        _simulation("a", 1, 1.0),
        _simulation("a", 2, 0.0, termination="infrastructure_error"),
    ]
    path.write_text(json.dumps(_metadata(simulations)), encoding="utf-8")
    receipt = aggregate_tau2_results(
        path,
        expected_cases=["retail/a@1", "retail/a@2"],
    )
    assert receipt["overall"]["eligible_simulations"] == 1
    assert receipt["overall"]["infrastructure_errors"] == 1
    assert receipt["status"] == "incomplete"


def test_tau2_result_rejects_duplicate_cases(tmp_path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(_metadata([_simulation("a", 1, 1.0), _simulation("a", 1, 1.0)])),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_tau2_results(path)
