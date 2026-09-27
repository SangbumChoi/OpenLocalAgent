#!/usr/bin/env python3
"""Run a bounded native tau2-bench domain probe without retaining task payloads.

This runner covers the local, resettable airline, retail, and telecom environments.  It executes
one LocalAgent turn per selected base task, replays the resulting calls through tau2's native
action/environment evaluators, and hashes task text, outputs, and source files.  It is deliberately
not a user-simulator or leaderboard runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from evaluate_tau2_mock_checkpoint import (
    _action_hashes,
    _gold_messages,
    _json_hash,
    _messages_from_calls,
    _sha256,
    _text_hash,
    _tool_registry,
)
from localagent.agent.runtime import Agent


DOMAINS = ("airline", "retail", "telecom")


def _source_files(root: Path, domain: str) -> dict[str, dict[str, Any]]:
    relative = ["LICENSE", "pyproject.toml", "README.md"]
    domain_root = root / "data" / "tau2" / "domains" / domain
    if not domain_root.is_dir():
        raise FileNotFoundError(f"tau2 domain directory is missing: {domain_root}")
    relative.extend(
        path.relative_to(root).as_posix()
        for path in sorted(domain_root.rglob("*"))
        if path.is_file()
    )
    return {
        name: _sha256(root / name)
        for name in relative
        if (root / name).is_file()
    }


def _domain_api(domain: str):
    if domain not in DOMAINS:
        raise ValueError(f"unsupported tau2 domain: {domain!r}")
    module = importlib.import_module(f"tau2.domains.{domain}.environment")
    return module.get_environment, module.get_tasks


def evaluate(
    *,
    root: Path,
    checkpoint: Path,
    domain: str,
    task_ids: list[str],
    report: Path,
    source_revision: str,
    retrieve_k: int = 50,
    selector_mode: str = "checkpoint",
    selector_first: bool = True,
    contract_task_id: str | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not task_ids:
        raise ValueError("at least one --task-id is required")
    if selector_mode not in {"checkpoint", "retriever"}:
        raise ValueError(f"unsupported selector mode: {selector_mode!r}")
    try:
        get_environment, get_tasks = _domain_api(domain)
        from tau2.evaluator.evaluator_action import ActionEvaluator
        from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    except ImportError as error:
        raise RuntimeError("install tau2-bench in an isolated environment before running") from error

    all_tasks = {task.id: task for task in get_tasks("base")}
    missing = sorted(set(task_ids) - set(all_tasks))
    if missing:
        raise ValueError(f"unknown tau2 {domain} base task IDs: {missing}")
    source = {
        "dataset": "tau2-bench",
        "domain": domain,
        "split": "base",
        "revision": source_revision,
        "source_url": "https://github.com/sierra-research/tau2-bench",
        "license": "MIT",
        "files": _source_files(root, domain),
        "task_ids": sorted(all_tasks),
        "task_count": len(all_tasks),
    }

    initial_environment = get_environment()
    state: dict[str, Any] = {"environment": initial_environment, "calls": []}
    registry = _tool_registry(state, initial_environment)
    agent = Agent.from_checkpoint(
        checkpoint,
        registry,
        selector_first=selector_first,
        retrieve_k=max(1, retrieve_k),
    )
    if selector_mode == "retriever":
        agent.selector = None
        agent.route_head = None

    contract_id = contract_task_id or task_ids[0]
    if contract_id not in all_tasks:
        raise ValueError(f"unknown tau2 contract task ID: {contract_id}")
    contract_task = all_tasks[contract_id]
    contract_environment = get_environment()
    try:
        contract_reward = EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=contract_task,
            full_trajectory=_gold_messages(contract_task, contract_environment),
            strict_replay=True,
        )
        contract_value = float(contract_reward.reward)
        contract_error = None
    except Exception as error:  # fail closed on upstream data drift
        contract_value = 0.0
        contract_error = repr(error)

    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = all_tasks[task_id]
        state["environment"] = get_environment()
        state["calls"] = []
        instruction = str(task.user_scenario.instructions)
        output = agent.chat(instruction)
        calls = state["calls"]
        messages = _messages_from_calls(calls, output)
        try:
            environment_result = EnvironmentEvaluator.calculate_reward(
                environment_constructor=get_environment,
                task=task,
                full_trajectory=messages,
                strict_replay=True,
            )
            environment_value = float(environment_result.reward)
            environment_error = None
        except Exception as error:  # fail closed on data/assertion drift
            environment_value = 0.0
            environment_error = repr(error)
        action_result = ActionEvaluator.calculate_reward(task=task, full_trajectory=messages)
        predicted = [
            {"name": item["call"].name, "arguments": item["call"].arguments}
            for item in calls
        ]
        expected = task.evaluation_criteria.actions if task.evaluation_criteria else []
        expected_count = len(expected or [])
        first_exact = bool(
            expected
            and calls
            and expected[0].compare_with_tool_call(calls[0]["call"])
        )
        record = {
            "task_id": task_id,
            "instruction": _text_hash(instruction),
            "model_output": _text_hash(output),
            "expected_actions": _action_hashes(task),
            "predicted_actions": {"count": len(predicted), "sha256": _json_hash(predicted)["sha256"]},
            "model_tool_calls": len(calls),
            "first_action_exact": first_exact,
            "action_reward": float(action_result.reward),
            "environment_reward": environment_value,
            "bounded_native_success": bool(
                expected_count > 0
                and calls
                and action_result.reward >= 1.0
                and environment_value >= 1.0
            ),
        }
        if environment_error is not None:
            record["environment_error"] = _text_hash(environment_error)
        records.append(record)

    result: dict[str, Any] = {
        "kind": "localagent_tau2_domain_checkpoint_native_probe",
        "schema_version": 1,
        "checkpoint": _sha256(checkpoint),
        "runner": {
            "package": "tau2",
            "version": importlib.metadata.version("tau2"),
            "source_revision": source_revision,
            "root": str(root.resolve()),
        },
        "source": source,
        "configuration": {
            "domain": domain,
            "split": "base",
            "tasks": task_ids,
            "max_agent_turns": 1,
            "user_simulator": False,
            "retrieve_k": retrieve_k,
            "selector_first": selector_first,
            "selector_mode": selector_mode,
        },
        "environment": {
            "native_runtime_executed": True,
            "reset_per_task": True,
            "external_accounts": False,
            "external_services": False,
            "screenshots": False,
        },
        "contract_verification": {
            "task_id": str(contract_task.id),
            "gold_trajectory_replayed": True,
            "environment_reward": contract_value,
            "passed": bool(contract_value >= 1.0),
        },
        "tasks": records,
        "summary": {
            "tasks": len(records),
            "model_tool_calls": sum(item["model_tool_calls"] for item in records),
            "first_action_exact": sum(int(item["first_action_exact"]) for item in records),
            "bounded_native_successes": sum(int(item["bounded_native_success"]) for item in records),
            "bounded_native_success_rate": sum(int(item["bounded_native_success"]) for item in records)
            / len(records),
        },
        "claim_boundary": (
            "Native resettable tau2-bench domain diagnostic with one LocalAgent turn per task and an "
            "oracle contract replay. This is not a complete domain split, user-simulator run, "
            "leaderboard score, external-service result, or WebGPU deployment claim. Task text, tool "
            "outputs, and checkpoint internals are hash-only."
        ),
    }
    if contract_error is not None:
        result["contract_verification"]["error"] = _text_hash(contract_error)
    result["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--selector-mode", choices=("checkpoint", "retriever"), default="checkpoint")
    parser.add_argument("--selector-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contract-task-id")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.task_ids:
        task_ids = args.task_ids
    else:
        _, get_tasks = _domain_api(args.domain)
        if args.max_tasks < 1:
            raise ValueError("--max-tasks must be positive")
        task_ids = [task.id for task in get_tasks("base")[: args.max_tasks]]
    result = evaluate(
        root=args.root,
        checkpoint=args.checkpoint,
        domain=args.domain,
        task_ids=task_ids,
        report=args.report,
        source_revision=args.source_revision,
        retrieve_k=args.retrieve_k,
        selector_mode=args.selector_mode,
        selector_first=args.selector_first,
        contract_task_id=args.contract_task_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
