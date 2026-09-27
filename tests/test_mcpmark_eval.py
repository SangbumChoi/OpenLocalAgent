import json

import pytest

from localagent.eval.mcpmark import aggregate_mcpmark_results, discover_mcpmark_tasks


def _write_result(root, service, run, task, success, tokens=10):
    path = root / f"model__{service}" / f"run-{run}" / task / "meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "execution_result": {"success": success},
                "agent_execution_time": 1.5,
                "turn_count": 2,
                "token_usage": {"total_tokens": tokens},
            }
        ),
        encoding="utf-8",
    )


def test_discover_mcpmark_tasks_uses_paths_only(tmp_path) -> None:
    path = tmp_path / "tasks/notion/standard/pages/create/meta.json"
    path.parent.mkdir(parents=True)
    path.write_text("this file is never opened", encoding="utf-8")
    assert discover_mcpmark_tasks(tmp_path) == {"notion": ("pages__create",)}


def test_aggregate_mcpmark_results_reports_pass_k_and_pass_power_k(tmp_path) -> None:
    expected = {"notion": ("pages__create", "pages__update")}
    _write_result(tmp_path, "notion", 1, "pages__create", True)
    _write_result(tmp_path, "notion", 1, "pages__update", False)
    _write_result(tmp_path, "notion", 2, "pages__create", False)
    _write_result(tmp_path, "notion", 2, "pages__update", True)
    result = aggregate_mcpmark_results(tmp_path, model="model", expected_tasks=expected, k=2)
    assert result["overall"]["pass_at_1"] == 0.5
    assert result["overall"]["pass_at_k"] == 1.0
    assert result["overall"]["pass_power_k"] == 0.0
    assert result["by_service"]["notion"]["run_success_rates"] == [0.5, 0.5]


def test_aggregate_mcpmark_results_fails_closed_on_missing_task(tmp_path) -> None:
    _write_result(tmp_path, "notion", 1, "pages__create", True)
    with pytest.raises(ValueError, match="meta.json"):
        aggregate_mcpmark_results(
            tmp_path,
            model="model",
            expected_tasks={"notion": ("pages__create", "pages__update")},
            k=1,
        )


def test_aggregate_mcpmark_results_weights_first_run_across_services(tmp_path) -> None:
    _write_result(tmp_path, "notion", 1, "pages__create", True)
    _write_result(tmp_path, "filesystem", 1, "files__read", False)
    result = aggregate_mcpmark_results(
        tmp_path,
        model="model",
        expected_tasks={"notion": ("pages__create",), "filesystem": ("files__read",)},
        k=1,
    )
    assert result["overall"]["pass_at_1"] == 0.5


def test_aggregate_mcpmark_results_accepts_easy_service_suffix(tmp_path) -> None:
    _write_result(tmp_path, "filesystem-easy", 1, "files__read", True)
    result = aggregate_mcpmark_results(
        tmp_path,
        model="model",
        expected_tasks={"filesystem": ("files__read",)},
        k=1,
    )
    assert result["by_service"]["filesystem"]["pass_at_1"] == 1.0
