"""Fail-closed aggregation for locally produced MCPMark result directories.

This module does not launch MCP servers or claim an official leaderboard score.  It consumes only
the result ``meta.json`` files emitted by MCPMark after an operator has run its pinned environment,
and requires a complete expected task map before calculating pass@k/pass^k-style metrics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SERVICE_ALIASES = {"playwright_webarena": "playwright", "supabase": "postgres", "insforge": "postgres"}


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _canonical_service(name: str) -> str:
    # MCPMark appends the selected task-suite to result directory names (for example,
    # ``model__filesystem-easy``), while task discovery is keyed by the base service.
    for suffix in ("-easy", "-standard"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _SERVICE_ALIASES.get(name, name)


def discover_mcpmark_tasks(checkout: str | Path, *, suite: str = "standard") -> dict[str, tuple[str, ...]]:
    """Discover expected task IDs from paths only; descriptions and metadata are not opened."""

    suite_text = _text(suite, label="suite")
    root = Path(checkout) / "tasks"
    if not root.is_dir():
        raise ValueError(f"MCPMark checkout is missing {root}")
    tasks: dict[str, list[str]] = {}
    for meta_path in sorted(root.glob(f"*/{suite_text}/*/*/meta.json")):
        parts = meta_path.relative_to(root).parts
        if len(parts) != 5:
            raise ValueError(f"unexpected MCPMark task path: {meta_path}")
        service, _, category, task, _ = parts
        canonical = _canonical_service(service)
        tasks.setdefault(canonical, []).append(f"{category}__{task}")
    if not tasks:
        raise ValueError(f"MCPMark checkout contains no {suite_text!r} tasks")
    return {service: tuple(sorted(values)) for service, values in sorted(tasks.items())}


def _read_result(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid MCPMark result metadata: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"MCPMark result must be an object: {path}")
    execution = value.get("execution_result")
    if not isinstance(execution, Mapping) or not isinstance(execution.get("success"), bool):
        raise ValueError(f"MCPMark result lacks boolean execution_result.success: {path}")
    return value


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_mcpmark_results(
    result_root: str | Path,
    *,
    model: str,
    expected_tasks: Mapping[str, Sequence[str]],
    k: int,
) -> dict[str, Any]:
    """Aggregate complete MCPMark results and fail closed on missing task/run records."""

    root = Path(result_root)
    model_name = _text(model, label="model")
    run_count = _positive_int(k, label="k")
    if not root.is_dir():
        raise ValueError(f"MCPMark result root is missing: {root}")
    if not expected_tasks:
        raise ValueError("expected_tasks must contain at least one service")
    expected: dict[str, tuple[str, ...]] = {}
    for service, tasks in expected_tasks.items():
        canonical = _canonical_service(_text(service, label="expected service"))
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
            raise ValueError(f"expected_tasks[{service!r}] must be a non-empty sequence")
        normalized = tuple(sorted(_text(task, label=f"expected_tasks[{service}][]") for task in tasks))
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"expected_tasks[{service!r}] contains duplicates")
        if canonical in expected:
            raise ValueError(f"duplicate expected service after aliasing: {canonical}")
        expected[canonical] = normalized

    model_dirs: dict[str, list[Path]] = {}
    for path in sorted(root.glob(f"{model_name}__*")):
        if path.is_dir():
            raw_service = path.name.split("__", 1)[1]
            model_dirs.setdefault(_canonical_service(raw_service), []).append(path)

    by_service: dict[str, dict[str, Any]] = {}
    all_successes: list[bool] = []
    all_first_run_successes: list[bool] = []
    all_times: list[float] = []
    all_turns: list[float] = []
    all_tokens: list[float] = []
    for service, tasks in sorted(expected.items()):
        dirs = model_dirs.get(service, [])
        if len(dirs) != 1:
            raise ValueError(f"expected exactly one result directory for {service!r}, found {len(dirs)}")
        service_dir = dirs[0]
        task_successes: dict[str, list[bool]] = {task: [] for task in tasks}
        run_rates: list[float] = []
        service_times: list[float] = []
        service_turns: list[float] = []
        service_tokens: list[float] = []
        for run_index in range(1, run_count + 1):
            run_dir = service_dir / f"run-{run_index}"
            if not run_dir.is_dir():
                raise ValueError(f"missing MCPMark run directory: {run_dir}")
            run_successes: list[bool] = []
            for task in tasks:
                meta = _read_result(run_dir / task / "meta.json")
                success = bool(meta["execution_result"]["success"])
                task_successes[task].append(success)
                run_successes.append(success)
                execution_time = meta.get("agent_execution_time", 0.0)
                turns = meta.get("turn_count", 0)
                token_usage = meta.get("token_usage", {})
                if not isinstance(execution_time, (int, float)) or not isinstance(turns, (int, float)):
                    raise ValueError(f"invalid timing/turn fields in {run_dir / task / 'meta.json'}")
                if not isinstance(token_usage, Mapping):
                    raise ValueError(f"invalid token_usage in {run_dir / task / 'meta.json'}")
                tokens = token_usage.get("total_tokens", 0)
                if not isinstance(tokens, (int, float)):
                    raise ValueError(f"invalid token_usage.total_tokens in {run_dir / task / 'meta.json'}")
                service_times.append(float(execution_time))
                service_turns.append(float(turns))
                service_tokens.append(float(tokens))
            run_rates.append(_mean([float(value) for value in run_successes]))
            if run_index == 1:
                all_first_run_successes.extend(run_successes)
        task_any = [any(values) for values in task_successes.values()]
        task_all = [all(values) for values in task_successes.values()]
        service_successes = [value for values in task_successes.values() for value in values]
        all_successes.extend(service_successes)
        all_times.extend(service_times)
        all_turns.extend(service_turns)
        all_tokens.extend(service_tokens)
        by_service[service] = {
            "tasks": len(tasks),
            "runs": run_count,
            "pass_at_1": run_rates[0],
            "pass_at_k": _mean([float(value) for value in task_any]),
            "pass_power_k": _mean([float(value) for value in task_all]),
            "run_success_rates": run_rates,
            "mean_agent_execution_time": _mean(service_times),
            "mean_turn_count": _mean(service_turns),
            "mean_total_tokens": _mean(service_tokens),
        }

    total_tasks = sum(len(tasks) for tasks in expected.values())
    total_runs = total_tasks * run_count
    return {
        "kind": "localagent_mcpmark_result_aggregate",
        "schema_version": 1,
        "model": model_name,
        "tasks": total_tasks,
        "runs": run_count,
        "overall": {
            "pass_at_1": sum(all_first_run_successes) / total_tasks,
            "pass_at_k": _mean(
                [
                    float(
                        any(
                            _read_result(
                                model_dirs[service][0]
                                / f"run-{run_index}"
                                / task
                                / "meta.json"
                            )["execution_result"]["success"]
                            for run_index in range(1, run_count + 1)
                        )
                    )
                    for service, tasks in sorted(expected.items())
                    for task in tasks
                ]
            ),
            "pass_power_k": _mean(
                [
                    float(
                        all(
                            _read_result(
                                model_dirs[service][0]
                                / f"run-{run_index}"
                                / task
                                / "meta.json"
                            )["execution_result"]["success"]
                            for run_index in range(1, run_count + 1)
                        )
                    )
                    for service, tasks in sorted(expected.items())
                    for task in tasks
                ]
            ),
            "mean_agent_execution_time": _mean(all_times),
            "mean_turn_count": _mean(all_turns),
            "mean_total_tokens": _mean(all_tokens),
            "run_records": total_runs,
        },
        "by_service": by_service,
        "claim_scope": "Complete local MCPMark result aggregation; not an official leaderboard score.",
    }


__all__ = ["aggregate_mcpmark_results", "discover_mcpmark_tasks"]
