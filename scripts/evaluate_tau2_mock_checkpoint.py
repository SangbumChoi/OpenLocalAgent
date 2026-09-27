#!/usr/bin/env python3
"""Run a bounded native τ-Bench mock-domain probe.

The optional tau2-bench dependency is loaded only when this command executes.  A public mock
task is reset, the current LocalAgent checkpoint is exposed to the real tau2 tool schemas, and
the resulting tool calls are replayed through tau2's independent environment evaluator.  The
first task is also replayed from its public reference actions to prove that the native verifier is
live.  Task text, tool outputs, and checkpoint internals are hashed rather than retained.

This is a native resettable diagnostic, not a complete tau2 leaderboard run: it uses one agent
turn per task, no user simulator, and no external account or service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.schema import ToolSpec


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _text_hash(value: str) -> dict[str, int | str]:
    encoded = value.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _json_hash(value: Any) -> dict[str, int | str]:
    return _text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _source_files(root: Path) -> dict[str, dict[str, Any]]:
    relative = (
        "LICENSE",
        "pyproject.toml",
        "README.md",
        "data/tau2/domains/mock/tasks.json",
        "data/tau2/domains/mock/split_tasks.json",
        "data/tau2/domains/mock/db.json",
        "data/tau2/domains/mock/user_db.json",
        "data/tau2/domains/mock/policy.md",
    )
    files: dict[str, dict[str, Any]] = {}
    for name in relative:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"tau2 source file is missing: {path}")
        files[name] = _sha256(path)
    return files


def _tool_registry(state: dict[str, Any], environment: Any) -> ToolRegistry:
    """Expose the native tau2 assistant tools through the LocalAgent registry."""

    from tau2.environment.toolkit import get_tool_signatures

    signatures = get_tool_signatures(environment.tools)
    registry = ToolRegistry()
    for tool in environment.get_tools():
        signature = signatures[tool.name]
        spec = ToolSpec(
            name=tool.name,
            description=tool.short_desc or tool.name,
            parameters=dict(signature.params),
        )

        def dispatch(name: str):
            def call(**kwargs: Any) -> Any:
                from tau2.data_model.message import ToolCall

                tool_call = ToolCall(
                    id=f"localagent-{len(state['calls']):03d}",
                    name=name,
                    arguments=kwargs,
                )
                response = state["environment"].get_response(tool_call)
                state["calls"].append({"call": tool_call, "response": response})
                try:
                    return json.loads(response.content or "null")
                except json.JSONDecodeError:
                    return response.content

            return call

        registry.register(spec, dispatch(tool.name))
    return registry


def _messages_from_calls(calls: list[dict[str, Any]], output: str):
    from tau2.data_model.message import AssistantMessage

    messages = []
    if calls:
        for item in calls:
            messages.extend(
                [
                    AssistantMessage(
                        role="assistant",
                        content="",
                        tool_calls=[item["call"]],
                    ),
                    item["response"],
                ]
            )
    else:
        messages.append(AssistantMessage(role="assistant", content=output))
    return messages


def _gold_messages(task: Any, environment: Any):
    from tau2.data_model.message import AssistantMessage, ToolCall

    messages = []
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
    for index, action in enumerate(actions or []):
        call = ToolCall(
            id=f"gold-{index:03d}",
            name=action.name,
            arguments=dict(action.arguments),
            requestor=action.requestor,
        )
        messages.extend(
            [
                AssistantMessage(role="assistant", content="", tool_calls=[call]),
                environment.get_response(call),
            ]
        )
    return messages


def _action_hashes(task: Any) -> dict[str, Any]:
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
    return {
        "count": len(actions or []),
        "sha256": _json_hash(
            [
                {
                    "name": action.name,
                    "arguments": action.arguments,
                    "requestor": action.requestor,
                }
                for action in actions or []
            ]
        )["sha256"],
    }


def evaluate(
    *,
    root: Path,
    checkpoint: Path,
    task_ids: list[str],
    report: Path,
    source_revision: str,
    retrieve_k: int = 50,
    selector_mode: str = "checkpoint",
    selector_first: bool = True,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not task_ids:
        raise ValueError("at least one --task-id is required")
    try:
        from tau2.domains.mock.environment import get_environment, get_tasks
        from tau2.evaluator.evaluator_action import ActionEvaluator
        from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    except ImportError as error:
        raise RuntimeError(
            "tau2-bench is optional; install it in an isolated environment before running this "
            "script (see docs/REALISTIC_AGENT_RESEARCH.md)."
        ) from error

    all_tasks = {task.id: task for task in get_tasks("base")}
    missing = sorted(set(task_ids) - set(all_tasks))
    if missing:
        raise ValueError(f"unknown tau2 mock base task IDs: {missing}")
    source = {
        "dataset": "tau2-bench",
        "domain": "mock",
        "split": "base",
        "revision": source_revision,
        "source_url": "https://github.com/sierra-research/tau2-bench",
        "license": "MIT",
        "files": _source_files(root),
        "task_ids": sorted(all_tasks),
        "task_count": len(all_tasks),
    }

    # Load the model once; the registry closures below switch to a freshly reset environment.
    initial_environment = get_environment()
    state: dict[str, Any] = {"environment": initial_environment, "calls": []}
    registry = _tool_registry(state, initial_environment)
    if selector_mode not in {"checkpoint", "retriever"}:
        raise ValueError(f"unsupported selector mode: {selector_mode!r}")
    agent = Agent.from_checkpoint(
        checkpoint,
        registry,
        selector_first=selector_first,
        retrieve_k=max(1, retrieve_k),
    )
    if selector_mode == "retriever":
        # A learned selector is trained over the public LocalAgent catalog.  For an unseen
        # benchmark schema, compare it with the zero-training name/description retriever instead
        # of silently treating the learned closed-world prior as generalization.
        agent.selector = None
        agent.route_head = None

    # A full public reference trajectory proves the resettable native evaluator without using
    # model output.  It is kept only as aggregate metrics in the receipt.
    contract_task = all_tasks[task_ids[0]]
    contract_environment = get_environment()
    try:
        contract_reward = EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=contract_task,
            full_trajectory=_gold_messages(contract_task, contract_environment),
            strict_replay=True,
        )
        contract_reward_value = float(contract_reward.reward)
        contract_error = None
    except Exception as error:  # the contract must fail closed if upstream data drifts
        contract_reward_value = 0.0
        contract_error = repr(error)

    task_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = all_tasks[task_id]
        state["environment"] = get_environment()
        state["calls"] = []
        instruction = str(task.user_scenario.instructions)
        output = agent.chat(instruction)
        calls = state["calls"]
        messages = _messages_from_calls(calls, output)
        try:
            env_reward = EnvironmentEvaluator.calculate_reward(
                environment_constructor=get_environment,
                task=task,
                full_trajectory=messages,
                strict_replay=True,
            )
            environment_reward = float(env_reward.reward)
            environment_error = None
        except Exception as error:  # fail closed on assertion/data drift, never claim a pass
            environment_reward = 0.0
            environment_error = repr(error)
        action_reward = ActionEvaluator.calculate_reward(task=task, full_trajectory=messages)
        predicted = [
            {"name": item["call"].name, "arguments": item["call"].arguments}
            for item in calls
        ]
        expected = task.evaluation_criteria.actions if task.evaluation_criteria else []
        expected_count = len(expected or [])
        first_action_exact = bool(
            expected
            and calls
            and expected[0].compare_with_tool_call(calls[0]["call"])
        )
        task_records.append(
            {
                "task_id": task_id,
                "instruction": _text_hash(instruction),
                "model_output": _text_hash(output),
                "expected_actions": _action_hashes(task),
                "predicted_actions": {"count": len(predicted), "sha256": _json_hash(predicted)["sha256"]},
                "model_tool_calls": len(calls),
                "first_action_exact": first_action_exact,
                "action_reward": float(action_reward.reward),
                "environment_reward": environment_reward,
                "bounded_native_success": bool(
                    expected_count > 0
                    and len(calls) > 0
                    and action_reward.reward >= 1.0
                    and environment_reward >= 1.0
                ),
            }
        )
        if environment_error is not None:
            task_records[-1]["environment_error"] = _text_hash(environment_error)

    result: dict[str, Any] = {
        "kind": "localagent_tau2_mock_checkpoint_native_probe",
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
            "domain": "mock",
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
            "task_id": contract_task.id,
            "gold_trajectory_replayed": True,
            "environment_reward": contract_reward_value,
            "passed": bool(contract_reward_value >= 1.0),
        },
        "tasks": task_records,
        "summary": {
            "tasks": len(task_records),
            "model_tool_calls": sum(item["model_tool_calls"] for item in task_records),
            "first_action_exact": sum(int(item["first_action_exact"]) for item in task_records),
            "bounded_native_successes": sum(
                int(item["bounded_native_success"]) for item in task_records
            ),
            "bounded_native_success_rate": sum(
                int(item["bounded_native_success"]) for item in task_records
            )
            / len(task_records),
        },
        "claim_boundary": (
            "Native tau2-bench mock-domain reset/evaluation with one LocalAgent turn per task and "
            "an oracle contract replay. This is not the complete tau2 base split, does not run the "
            "user simulator, and is not a tau2 leaderboard, retail/telecom, email, or external "
            "service score. Task text, tool outputs, and checkpoint internals are hash-only."
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
    parser.add_argument("--root", type=Path, required=True, help="tau2-bench source checkout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument(
        "--selector-mode",
        choices=("checkpoint", "retriever"),
        default="checkpoint",
        help="use the checkpoint selector or the zero-training schema retriever",
    )
    parser.add_argument(
        "--selector-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="emit the first selected grounded body instead of model-ranking candidates",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.task_ids:
        task_ids = args.task_ids
    else:
        from tau2.domains.mock.environment import get_tasks

        if args.max_tasks < 1:
            raise ValueError("--max-tasks must be positive")
        task_ids = [task.id for task in get_tasks("base")[: args.max_tasks]]
    result = evaluate(
        root=args.root,
        checkpoint=args.checkpoint,
        task_ids=task_ids,
        report=args.report,
        source_revision=args.source_revision,
        retrieve_k=args.retrieve_k,
        selector_mode=args.selector_mode,
        selector_first=args.selector_first,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
