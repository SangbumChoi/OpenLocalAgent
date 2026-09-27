#!/usr/bin/env python3
"""Run the WebGPU checkpoint over MobileGym's complete public test split.

This runner deliberately keeps the deployment contract visible: the pinned MobileGym
simulator, official test whitelist, task setup, action handlers, and state-diff judge are
executed unchanged.  The small model receives a bounded textual DOM projection rather than a
screenshot, so the receipt is a *native text-first MobileGym evaluation*, not a visual-agent
leaderboard claim.  Task instructions, sampled values, screenshots, and raw model arguments are
never written to the receipt; only public task IDs, hashes, action names, and aggregate outcomes
are retained.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MOBILEGYM_URL = "https://github.com/Purewhiter/mobilegym"
DEFAULT_REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _split_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _compact_judge(judge: Any) -> dict[str, Any]:
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


def _task_summary(task_id: str, judge: Any, trace: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "suite": task_id.split(".", 1)[0],
        "passed": bool(judge.passed),
        "clean": bool(judge.clean),
        "progress": float(judge.progress),
        "steps": len(trace),
        "model_invocations": len(trace),
        "tool_names": sorted({str(item.get("tool")) for item in trace if item.get("tool")}),
        "trace_sha256": _sha256_json(trace),
        "judge": _compact_judge(judge),
        "elapsed_seconds": round(float(elapsed), 3),
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

    split_path = source_root / "bench_env" / "splits" / "test.txt"
    train_path = source_root / "bench_env" / "splits" / "train.txt"
    ordered_test_ids = _split_ids(split_path)
    train_ids = set(_split_ids(train_path))
    test_ids = set(ordered_test_ids)
    resolved_test = resolve_split("test")
    if test_ids != resolved_test or len(test_ids) != len(ordered_test_ids):
        raise RuntimeError("MobileGym test split is not a unique, resolver-equivalent whitelist")
    if train_ids & test_ids:
        raise RuntimeError("MobileGym train/test split overlap detected")

    all_tasks = load_tasks(seed=args.seed, sample_templates=True)
    task_by_id = {task.id: task for task in all_tasks}
    missing = sorted(test_ids - set(task_by_id))
    if missing:
        raise RuntimeError(f"MobileGym test IDs missing from registry: {missing[:5]}")

    selected_ids = ordered_test_ids[args.start :]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    if not selected_ids:
        raise ValueError("selected MobileGym test slice is empty")

    registry = ToolRegistry()
    specs = mobile_tools()
    for spec in specs:
        registry.register(spec, lambda **kwargs: kwargs)
    agent = Agent.from_checkpoint(
        args.checkpoint,
        registry,
        selector_top_m=args.selector_top_m,
        selector_first=args.selector_first,
    )
    env = MobileGymEnv(
        url=args.env_url,
        headless=True,
        delay_after_action=0.0,
        verbose=False,
        coord_space="norm_0_1000",
    )

    from scripts.run_mobilegym_model_probe import _action_from_call, _build_prompt

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        await env.start()
        for index, task_id in enumerate(selected_ids, start=args.start):
            task = task_by_id[task_id]
            trace: list[dict[str, Any]] = []
            episode_started = time.monotonic()
            try:
                init_obs = await task.setup(env)
                last_obs = init_obs
                for step in range(max(1, args.max_steps)):
                    body_text = await env.page.locator("body").inner_text()
                    route = await env.get_route()
                    prompt = _build_prompt(task, body_text, route)
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
                        trace.append(
                            {
                                "step": step,
                                "tool": None,
                                "model_output_sha256": _sha256_json(rendered),
                            }
                        )
                        break
                    call = calls[0]
                    action = _action_from_call(call)
                    trace.append(
                        {
                            "step": step,
                            "tool": call.name,
                            "argument_keys": sorted(str(key) for key in call.arguments),
                            "arguments_sha256": _sha256_json(call.arguments),
                            "model_output_sha256": _sha256_json(rendered),
                            "translated": action is not None,
                        }
                    )
                    if action is None:
                        break
                    result = await env.step(action)
                    last_obs = result.observation
                    if result.done:
                        break
                if last_obs is None:
                    last_obs = await env.get_observation()
                if not env._done:
                    await env.step(Action.complete("full_mobilegym_text_projection"))
                    last_obs = await env.get_observation()
                judge = task.evaluate(
                    JudgeInput(init_obs=init_obs, last_obs=last_obs, answer=env.agent_answer)
                )
                results.append(_task_summary(task_id, judge, trace, time.monotonic() - episode_started))
            except Exception as exc:  # keep the full-split run auditable and fail closed
                errors.append(
                    {
                        "task_id": task_id,
                        "error_type": type(exc).__name__,
                        "error_sha256": _sha256_json(str(exc)),
                    }
                )
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(
                    f"completed={index + 1}/{len(ordered_test_ids)} "
                    f"evaluated={len(results)} errors={len(errors)}",
                    flush=True,
                )
    finally:
        await env.close()

    passed = sum(1 for item in results if item["passed"])
    by_suite: dict[str, dict[str, Any]] = defaultdict(lambda: {"tasks": 0, "passed": 0})
    tool_counts: Counter[str] = Counter()
    for item in results:
        suite = str(item["suite"])
        by_suite[suite]["tasks"] += 1
        by_suite[suite]["passed"] += int(item["passed"])
        tool_counts.update(item["tool_names"])
    for summary in by_suite.values():
        summary["success_rate"] = summary["passed"] / summary["tasks"] if summary["tasks"] else 0.0

    full_split = args.start == 0 and args.limit is None and len(results) == len(ordered_test_ids)
    payload: dict[str, Any] = {
        "kind": "localagent_mobilegym_native_text_eval",
        "schema_version": 1,
        "benchmark_id": "mobilegym",
        "source": {
            "repository": MOBILEGYM_URL,
            "revision": args.revision,
            "source_root": source_root.name,
            "test_split_sha256": _sha256_file(split_path),
            "train_split_sha256": _sha256_file(train_path),
            "test_ids_sha256": _sha256_json(ordered_test_ids),
        },
        "environment_executed": True,
        "official_split": "test",
        "official_split_verified": bool(
            len(test_ids) == 256 and len(train_ids) == 160 and not (train_ids & test_ids)
        ),
        "native_receipt_eligible": bool(full_split and not errors),
        "task_count": len(results),
        "official_test_task_count": len(ordered_test_ids),
        "success_rate": passed / len(results) if results else 0.0,
        "passed_tasks": passed,
        "failed_tasks": len(results) - passed,
        "errors": errors,
        "run": {
            "seed": args.seed,
            "max_steps": max(1, args.max_steps),
            "selector_top_m": args.selector_top_m,
            "selector_first": args.selector_first,
            "start": args.start,
            "limit": args.limit,
            "full_official_test_split": full_split,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "observation_mode": "text_projection",
        "vision_used": False,
        "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
        "task_results": results,
        "suite_summary": dict(sorted(by_suite.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "claim_boundary": (
            "Native MobileGym simulator and official state-diff judge over the complete public "
            "test split using a bounded DOM/text observation projection. This is not a visual "
            "mobile-agent score, Android emulator result, or claim of screenshot grounding."
        ),
    }
    payload["receipt_self_sha256"] = _sha256_json(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--selector-top-m", type=int, default=1)
    parser.add_argument(
        "--selector-first",
        action="store_true",
        help="emit the top learned selector candidate without LM re-ranking it",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(_run(args))
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
