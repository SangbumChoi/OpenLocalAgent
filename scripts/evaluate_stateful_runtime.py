#!/usr/bin/env python
"""Evaluate a checkpoint in the resettable local email/Notion/browser runtime.

This is the deployment-shaped companion to ``train_stateful_productivity_probe.py``.  Unlike the
older fixed-step diagnostic, a rejected call does not advance the episode: the agent receives the
same state plus an error observation and may retry.  An oracle pass is run first to validate that
the runtime/verifier can reach 100%; the model pass then reports its real closed-loop result.

The runtime is deterministic and in-memory.  It is not an Android emulator, browser session,
MCP server, or real email/Notion account, and the receipt must not be presented as a public
benchmark score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector
from localagent.agent.parser import extract_tool_calls
from localagent.agent.routes import RouteHead
from localagent.data.stateful_productivity import (
    StatefulRuntime,
    build_tasks,
    canonical_json,
    stateful_reward,
    stateful_reward_spec,
    task_complete,
    tool_specs,
)

import train_stateful_productivity_probe as probe


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _state_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_heads(checkpoint: dict[str, Any], model, tools, device: str):
    """Restore the deployment heads stored by the stateful continuation."""

    route = RouteHead(model.cfg.d_model).to(device)
    route.load_state_dict(checkpoint["route_head"])
    dense = DenseToolSelector(
        model.cfg.d_model, proj=int(checkpoint.get("selector_proj", 256))
    ).to(device)
    dense.load_state_dict(checkpoint["dense_selector"])
    # Public WebGPU checkpoints before the stateful probe stored the legacy 17-argument pointer
    # vocabulary without ``ptr_args`` metadata.  Reuse the canonical migration helper so those
    # tensors load into the current 23-argument stateful vocabulary without a shape mismatch.
    pointer = probe._warm_pointer(checkpoint, model.cfg.d_model, random_init=False).to(device)
    selector = BoundSelector(
        dense,
        tools,
        device=device,
        examples=checkpoint.get("examples", {}),
    )
    return route.eval(), selector, pointer.eval()


def _oracle_call(runtime: StatefulRuntime) -> tuple[str | None, dict[str, Any]]:
    action = runtime.task.actions[runtime.step_index]
    return action.tool, copy.deepcopy(action.arguments)


def _model_call(
    runtime: StatefulRuntime,
    *,
    model,
    tokenizer,
    tools,
    route,
    selector,
    pointer,
    device,
    top_m: int,
    lexical_weight: float,
    blocked_candidates: set[str] | None = None,
    selector_first: bool = False,
):
    output = hybrid_decode(
        model,
        tokenizer,
        runtime.prompt(),
        tools,
        device=device,
        selector=selector,
        route_head=route,
        ptr_head=pointer,
        top_m=top_m,
        lexical_weight=lexical_weight,
        blocked_candidates=blocked_candidates,
        selector_first=selector_first,
    )
    calls = extract_tool_calls(output)
    if not calls:
        return None, {}, output
    call = calls[0]
    return call.name, dict(call.arguments), output


def _episode(
    task,
    policy: Callable[[StatefulRuntime], tuple[str | None, dict[str, Any]]],
    *,
    max_attempts_per_step: int,
    keep_output: bool = False,
) -> dict[str, Any]:
    runtime = StatefulRuntime(task)
    attempts = 0
    shaped = 0.0
    outputs: list[str] = []
    while not runtime.done and attempts < len(task.actions) * max_attempts_per_step:
        tool, arguments, *rest = policy(runtime)
        if keep_output and rest:
            outputs.append(rest[0])
        result = runtime.execute(tool, arguments)
        shaped += stateful_reward(result, terminal=runtime.done)
        attempts += 1
    events = runtime.events
    return {
        "task_id": task.task_id,
        "family": task.family,
        "attempts": attempts,
        "accepted_steps": sum(event["accepted"] for event in events),
        "expected_steps": len(task.actions),
        "task_complete": runtime.done and task_complete(task, runtime.state),
        "mean_shaped_reward": shaped / max(1, attempts),
        "event_sha256": _state_hash(events),
        "events": events,
        **({"model_outputs": outputs} if keep_output else {}),
    }


def _model_episode(
    task,
    *,
    model,
    tokenizer,
    tools,
    route,
    selector,
    pointer,
    device: str,
    top_m: int,
    max_attempts_per_step: int,
    selector_first: bool,
    lexical_weight: float,
) -> dict[str, Any]:
    """Run one model episode while remembering exact rejected outputs for retries."""

    blocked_candidates: set[str] = set()

    def policy(runtime: StatefulRuntime):
        if runtime.events and runtime.events[-1]["accepted"]:
            blocked_candidates.clear()
        result = _model_call(
            runtime,
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            route=route,
            selector=selector,
            pointer=pointer,
            device=device,
            top_m=top_m,
            lexical_weight=lexical_weight,
            blocked_candidates=blocked_candidates,
            selector_first=selector_first,
        )
        blocked_candidates.add(result[2])
        return result

    return _episode(
        task,
        policy,
        max_attempts_per_step=max_attempts_per_step,
        keep_output=True,
    )


def _aggregate(rows: list[dict[str, Any]], *, max_attempts_per_step: int) -> dict[str, Any]:
    totals = Counter()
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[row["family"]].append(row)
        totals["attempts"] += row["attempts"]
        totals["accepted_steps"] += row["accepted_steps"]
        totals["expected_steps"] += row["expected_steps"]
        totals["task_complete"] += int(row["task_complete"])
    return {
        "tasks": len(rows),
        "attempts": totals["attempts"],
        "accepted_steps": totals["accepted_steps"],
        "expected_steps": totals["expected_steps"],
        "attempt_success_rate": totals["accepted_steps"] / max(1, totals["attempts"]),
        "task_complete_rate": totals["task_complete"] / max(1, len(rows)),
        "recovery_task_complete_rate": sum(
            row["task_complete"] for row in rows if row["family"] == "recovery"
        ) / max(1, sum(row["family"] == "recovery" for row in rows)),
        "abstention_exact": sum(
            row["task_complete"] for row in rows if row["family"] == "abstention"
        ) / max(1, sum(row["family"] == "abstention" for row in rows)),
        "mean_shaped_reward": sum(row["mean_shaped_reward"] for row in rows) / max(1, len(rows)),
        "max_attempts_per_step": max_attempts_per_step,
        "by_family": {
            family: {
                "tasks": len(group),
                "attempts": sum(row["attempts"] for row in group),
                "accepted_steps": sum(row["accepted_steps"] for row in group),
                "expected_steps": sum(row["expected_steps"] for row in group),
                "task_complete_rate": sum(row["task_complete"] for row in group) / len(group),
            }
            for family, group in sorted(family_rows.items())
        },
        "task_rows": [
            {key: value for key, value in row.items() if key not in {"events", "model_outputs"}}
            for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-attempts-per-step", type=int, default=3)
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=0.5,
        help="blend weight for the current action-tail lexical selector query",
    )
    parser.add_argument(
        "--selector-first",
        action="store_true",
        help="choose the highest-ranked grounded candidate instead of LM reranking",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.max_attempts_per_step < 1:
        raise ValueError("max-attempts-per-step must be positive")
    if args.top_m < 1:
        raise ValueError("top-m must be positive")

    checkpoint, model, tokenizer = probe._load_parent(args.checkpoint)
    model.to(args.device).eval()
    tools = tool_specs()
    route, selector, pointer = _load_heads(checkpoint, model, tools, args.device)
    tasks = build_tasks("eval")

    oracle_rows = [
        _episode(
            task,
            _oracle_call,
            max_attempts_per_step=args.max_attempts_per_step,
        )
        for task in tasks
    ]
    model_rows = [
        _model_episode(
            task,
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            route=route,
            selector=selector,
            pointer=pointer,
            device=args.device,
            top_m=args.top_m,
            lexical_weight=args.lexical_weight,
            max_attempts_per_step=args.max_attempts_per_step,
            selector_first=args.selector_first,
        )
        for task in tasks
    ]
    checkpoint_bytes, checkpoint_sha = _sha256(args.checkpoint)
    report = {
        "kind": "localagent_stateful_runtime_evaluation",
        "schema_version": 1,
        "suite": "localagent-stateful-productivity-v1",
        "runtime": {
            "kind": "local_resettable_state_machine",
            "environment_executed": True,
            "external_accounts_used": False,
            "public_benchmark": False,
            "tool_side_effects": "in_memory_only",
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha,
            "stage": checkpoint.get("stage"),
            "config": checkpoint["cfg"].get("name"),
        },
        "configuration": {
            "device": args.device,
            "max_attempts_per_step": args.max_attempts_per_step,
            "top_m": args.top_m,
            "lexical_weight": args.lexical_weight,
            "selector_first": args.selector_first,
            "rejection_memory": "exact_decoder_output_per_episode",
            "tool_pool_size": len(tools),
            "tool_pool_sha256": _state_hash([tool.name for tool in tools]),
            "reward_spec": stateful_reward_spec(),
        },
        "oracle": _aggregate(oracle_rows, max_attempts_per_step=args.max_attempts_per_step),
        "model": _aggregate(model_rows, max_attempts_per_step=args.max_attempts_per_step),
        "oracle_event_sha256": _state_hash(oracle_rows),
        "model_event_sha256": _state_hash(model_rows),
        "claim_boundary": (
            "The oracle pass validates the local reset/retry/verifier contract and is expected to "
            "reach 100%. The model pass is checkpoint-in-loop evidence for this synthetic runtime. "
            "Neither pass is an AndroidWorld, BrowserGym, OSWorld, MCPMark, ToolSandbox, email, "
            "Notion, or native WebGPU benchmark result."
        ),
    }
    report["receipt_self_sha256"] = _state_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
