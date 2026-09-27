#!/usr/bin/env python3
"""Run a bounded LocalAgent probe against an official MobileGym test task.

This is deliberately a *probe*, not a publication receipt.  It executes the real
MobileGym browser environment and state judge, but keeps the current model's
text-first mobile boundary explicit: no screenshot is passed to the decoder and
only a compact DOM text projection is used.  The output omits task instructions,
state values, and tool arguments; it is suitable for a reproducible failure
record without leaking benchmark content into the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    return None


def _action_from_call(call: Any):
    """Translate the additive mobile tool schema to MobileGym Action objects."""

    from bench_env.env.base import Action, ActionType

    name = str(call.name)
    args = dict(call.arguments)
    if name == "mobile_click":
        x, y = _normalized_number(args.get("x")), _normalized_number(args.get("y"))
        return Action.click([x, y]) if x is not None and y is not None else None
    if name == "mobile_long_press":
        x, y = _normalized_number(args.get("x")), _normalized_number(args.get("y"))
        return (
            Action(ActionType.LONG_PRESS, {"point": [x, y]})
            if x is not None and y is not None
            else None
        )
    if name == "mobile_swipe":
        keys = ("start_x", "start_y", "end_x", "end_y")
        values = [_normalized_number(args.get(key)) for key in keys]
        if all(value is not None for value in values):
            return Action.swipe(values[:2], values[2:])
        return None
    if name == "mobile_scroll":
        direction = str(args.get("direction", "")).lower()
        points = {
            "up": ([500, 750], [500, 250]),
            "down": ([500, 250], [500, 750]),
            "left": ([750, 500], [250, 500]),
            "right": ([250, 500], [750, 500]),
        }
        if direction in points:
            start, end = points[direction]
            return Action.swipe(start, end)
        return None
    if name == "mobile_open_app":
        app_name = str(args.get("app_name", "")).strip()
        return Action.awake(app_name) if app_name else None
    if name == "mobile_input_text":
        return Action.type_text(str(args.get("text", "")))
    if name == "mobile_navigate_home":
        return Action.home()
    if name == "mobile_navigate_back":
        return Action.back()
    if name == "mobile_press_enter":
        return Action(ActionType.ENTER, {})
    if name == "mobile_wait":
        seconds = args.get("seconds", 1.0)
        try:
            return Action.wait(float(seconds))
        except (TypeError, ValueError):
            return None
    if name == "mobile_submit_answer":
        return Action.answer(str(args.get("message", "")))
    return None


def _build_prompt(task: Any, body_text: str, route: dict[str, Any]) -> str:
    # Keep the text projection bounded so a large simulator state cannot dominate
    # a small WebGPU context.  The raw text never leaves this process.
    compact = " ".join(body_text.split())[:6000]
    return (
        f"Task: {task.description}\n"
        f"Current mobile app: {route.get('app', '')}; path: {route.get('path', '')}\n"
        f"Visible text: {compact}\n"
        "Choose one mobile action or abstain."
    )


def _compact_judge(judge: Any) -> dict[str, Any]:
    """Keep judge outcome structure while dropping benchmark state values."""

    return {
        "success": bool(judge.success),
        "clean": bool(judge.clean),
        "progress": float(judge.progress),
        "passed": bool(judge.passed),
        "issue_count": len(judge.issues),
        "issue_fields": sorted(
            str(item.get("field", "")) for item in judge.issues if isinstance(item, dict)
        ),
        "warning_count": len(judge.warnings),
        "warning_fields": sorted(
            str(item.get("field", "")) for item in judge.warnings if isinstance(item, dict)
        ),
        "judge_error": bool(judge.judge_error),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root))

    from bench_env.env.base import Action
    from bench_env.env.mobile_gym import MobileGymEnv
    from bench_env.splits import resolve_split
    from bench_env.task import JudgeInput, load_tasks
    from localagent.agent.constrained import grounded_decode, hybrid_decode
    from localagent.agent.mobile_toolset import mobile_tools
    from localagent.agent.parser import extract_tool_calls
    from localagent.agent.runtime import Agent
    from localagent.agent.tools import ToolRegistry

    test_split = resolve_split("test")
    if args.task_id not in test_split:
        raise ValueError(f"task_id is not in MobileGym's official test split: {args.task_id}")

    registry = ToolRegistry()
    specs = mobile_tools()
    for spec in specs:
        registry.register(spec, lambda **kwargs: kwargs)
    agent = Agent.from_checkpoint(args.checkpoint, registry, selector_top_m=args.selector_top_m)
    env = MobileGymEnv(
        url=args.env_url,
        headless=True,
        delay_after_action=0.0,
        verbose=False,
        coord_space="norm_0_1000",
    )
    task = next(
        task
        for task in load_tasks(suite=args.task_id.split(".", 1)[0], seed=args.seed, sample_templates=True)
        if task.id == args.task_id
    )
    trace: list[dict[str, Any]] = []
    model_invocations = 0
    model_calls = 0
    init_obs = None
    last_obs = None
    try:
        await env.start()
        init_obs = await task.setup(env)
        last_obs = init_obs
        for step in range(max(1, args.max_steps)):
            body_text = await env.page.locator("body").inner_text()
            route = await env.get_route()
            prompt = _build_prompt(task, body_text, route)
            model_invocations += 1
            if agent.selector is not None:
                rendered = hybrid_decode(
                    agent.model,
                    agent.tokenizer,
                    prompt,
                    list(agent.catalog.values()),
                    selector=agent.selector,
                    route_head=agent.route_head,
                    ptr_head=agent.ptr_head,
                    top_m=agent.selector_top_m,
                    selector_first=agent.selector_first,
                )
            else:
                rendered = grounded_decode(
                    agent.model,
                    agent.tokenizer,
                    prompt,
                    list(agent.catalog.values()),
                    tool_head=agent.tool_head,
                    ptr_head=agent.ptr_head,
                )
            calls = extract_tool_calls(rendered)
            if not calls:
                trace.append({"step": step, "model_output_sha256": _sha256_json(rendered), "tool": None})
                break
            call = calls[0]
            model_calls += 1
            action = _action_from_call(call)
            event: dict[str, Any] = {
                "step": step,
                "tool": call.name,
                "argument_keys": sorted(str(key) for key in call.arguments),
                "arguments_sha256": _sha256_json(call.arguments),
                "model_output_sha256": _sha256_json(rendered),
                "translated": action is not None,
            }
            trace.append(event)
            if action is None:
                break
            result = await env.step(action)
            last_obs = result.observation
            if result.done:
                break
        if last_obs is None:
            last_obs = await env.get_observation()
        # End the bounded probe without adding an agent-generated terminal action.
        if not env._done:
            await env.step(Action.complete("bounded_model_probe"))
            # CompleteHandler returns a lightweight observation without state;
            # fetch the public full observation for a fair state judge.
            last_obs = await env.get_observation()
        judge = task.evaluate(JudgeInput(init_obs=init_obs, last_obs=last_obs, answer=env.agent_answer))
    finally:
        await env.close()

    payload = {
        "schema_version": 1,
        "kind": "localagent_mobilegym_model_probe",
        "benchmark_id": "mobilegym_model_probe",
        "native_benchmark_id": "mobilegym",
        "native_receipt_eligible": False,
        "environment_executed": True,
        "official_split": "test",
        "official_split_verified": True,
        "task_id": args.task_id,
        "task_count": 1,
        "success_rate": 1.0 if judge.passed else 0.0,
        "judge": _compact_judge(judge),
        "observation_mode": "text_projection",
        "vision_used": False,
        "max_steps": max(1, args.max_steps),
        "model_invocations": model_invocations,
        "model_calls": model_calls,
        "trace": trace,
        "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
        "source_root": source_root.name,
        "source_revision": args.revision,
    }
    payload["receipt_self_sha256"] = _sha256_json(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", default="notes.ReadTodoText")
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument(
        "--selector-top-m",
        type=int,
        default=1,
        help="number of learned selector candidates passed to grounded reranking",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = asyncio.run(_run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
