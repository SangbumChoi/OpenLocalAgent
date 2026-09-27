#!/usr/bin/env python
"""Train and evaluate a closed-loop email/Notion/browser stateful probe.

The probe uses only the local deterministic state machine in
``localagent.data.stateful_productivity``.  It compares a frozen pretrained WebGPU backbone with
a shape-matched random-backbone control, trains the same route/dense-selector/pointer budgets, and
executes each predicted action through the state machine.  It is deliberately not an AndroidWorld,
BrowserGym, OSWorld, MCPMark, or real-account score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.pointer_head import PTR_ARGS, PointerHead, gold_span
from localagent.agent.routes import ROUTE_INDEX, ROUTES, RouteHead, route_of
from localagent.agent.tool_head import _feat
from localagent.agent.parser import extract_tool_calls
from localagent.data.agent_synth import Sample
from localagent.data.stateful_productivity import (
    SUITE_ID,
    apply_action,
    build_tasks,
    canonical_json,
    state_prompt,
    suite_inventory,
    stateful_reward,
    stateful_reward_spec,
    task_complete,
    tool_specs,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer


STATEFUL_PTR_ARGS = [*PTR_ARGS, "to", "subject", "body", "app_name", "text", "target"]
STATEFUL_PTR_ARG_SET = set(STATEFUL_PTR_ARGS)


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


def _tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _relative_l2(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in before.items():
        if name not in after:
            continue
        left = value.detach().float()
        right = after[name].detach().float()
        numerator += float((right - left).pow(2).sum())
        denominator += float(left.pow(2).sum())
    return (numerator**0.5) / max(denominator**0.5, 1e-12)


def _load_parent(path: Path) -> tuple[dict[str, Any], LocalAgentLM, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["cfg"])
    cfg.assert_within_budget()
    model = LocalAgentLM(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = checkpoint.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(metadata.get("kind", "byte"), metadata.get("path"))
    return checkpoint, model, tokenizer


def _rows(tasks) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        state = copy.deepcopy(task.initial_state)
        for index, action in enumerate(task.actions):
            prompt = state_prompt(task, index, state)
            if action.tool is None:
                sample = Sample(
                    "stateful_productivity",
                    task.family,
                    prompt,
                    "text",
                    "I won't invoke a tool.",
                    "text",
                    "{}",
                )
            else:
                sample = Sample(
                    "stateful_productivity",
                    task.family,
                    prompt,
                    "tool",
                    json.dumps(
                        {"arguments": action.arguments, "name": action.tool},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    action.tool,
                    canonical_json(action.arguments),
                )
            rows.append({"task": task, "step": index, "state": copy.deepcopy(state), "sample": sample})
            if action.tool is not None:
                state = apply_action(task, index, state, action.tool, action.arguments).state
    return rows


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build deterministic query examples for the dense selector.

    Stateful prompts contain two different retrieval signals: the long-horizon goal/state
    context and the current action instruction.  Indexing only the full prompt makes the tool
    tower overfit to the goal (for example, an email episode retrieves ``email_send`` even while
    the next action is ``mobile_open_app``).  Keep both views so inference can match either a
    full conversation or an action-tail query without introducing tool-specific rules.
    """

    from localagent.agent.constrained import _action_tail

    examples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample = row["sample"]
        if sample.kind == "tool":
            examples[sample.ref_name].append(sample.prompt)
            tail = _action_tail(sample.prompt)
            if tail and tail != sample.prompt:
                examples[sample.ref_name].append(tail)
    return {name: list(dict.fromkeys(values)) for name, values in examples.items()}


_ACTION_REWRITES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Send ", ("Submit ", "Deliver ", "Dispatch ")),
    ("Select ", ("Tap ", "Click ")),
    ("Tap ", ("Select ", "Click ")),
    ("Type ", ("Enter ", "Input ")),
    ("Put ", ("Enter ", "Insert ")),
    ("Fill ", ("Enter ", "Input ")),
    ("Open ", ("Navigate to ", "Bring up ")),
    ("Press ", ("Hit ",)),
    ("Save ", ("Store ",)),
    ("Create ", ("Make ",)),
)


def _action_augmented_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add deterministic, slot-preserving action paraphrases to the training view.

    The stateful benchmark deliberately keeps train/eval wording disjoint.  A tiny model still
    benefits from seeing a few generic UI synonyms (``Select``/``Tap``, ``Send``/``Submit``), so
    augment only the training prompts and leave task IDs, arguments, and evaluation rows intact.
    This is lexical data augmentation, not a tool-specific runtime rule.
    """

    augmented = list(rows)
    for row in rows:
        prompt = row["sample"].prompt
        marker = "Next required action:"
        if marker not in prompt:
            continue
        prefix, tail = prompt.split(marker, 1)
        action, separator, suffix = tail.partition(" Last tool result:")
        leading = action[: len(action) - len(action.lstrip())]
        action = action.strip()
        for source, replacements in _ACTION_REWRITES:
            if not action.startswith(source):
                continue
            for replacement in replacements:
                variant = f"{prefix}{marker}{leading}{replacement}{action[len(source):]}"
                if separator:
                    variant += f"{separator}{suffix}"
                augmented.append(
                    {
                        **row,
                        "sample": replace(row["sample"], prompt=variant),
                    }
                )
            break
    return augmented


def _features(model: LocalAgentLM, tokenizer, rows: list[dict[str, Any]], device: str) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [
                _feat(
                    model,
                    tokenizer,
                    row["sample"].prompt,
                    device,
                ).detach()
                for row in rows
            ]
        )


def _warm_pointer(parent: dict[str, Any], d_model: int, *, random_init: bool) -> PointerHead:
    pointer = PointerHead(d_model, args=STATEFUL_PTR_ARGS)
    if random_init or not parent.get("ptr_head"):
        return pointer
    legacy_args = parent.get("ptr_args", PTR_ARGS)
    legacy = PointerHead(d_model, args=legacy_args)
    legacy.load_state_dict(parent["ptr_head"])
    state = pointer.state_dict()
    old = legacy.state_dict()
    for index, name in enumerate(legacy.args):
        if name in pointer.arg_idx:
            state["arg_emb.weight"][pointer.arg_idx[name]] = old["arg_emb.weight"][index]
    state["start.weight"] = old["start.weight"]
    state["end.weight"] = old["end.weight"]
    pointer.load_state_dict(state)
    return pointer


def _train_pointer(
    model: LocalAgentLM,
    tokenizer,
    rows: list[dict[str, Any]],
    pointer: PointerHead,
    *,
    steps: int,
    seed: int,
    device: str,
) -> None:
    examples: list[tuple[torch.Tensor, int, int, int]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            sample = row["sample"]
            if sample.kind != "tool":
                continue
            arguments = json.loads(sample.ref_args)
            ids = tokenizer.encode(f"{USER}{sample.prompt}{ASSISTANT}")
            _, hidden = model(torch.tensor([ids], dtype=torch.long, device=device), return_hidden=True)
            for name, value in arguments.items():
                if name not in STATEFUL_PTR_ARG_SET or not isinstance(value, str):
                    continue
                value_ids = tokenizer.encode(value)
                span = gold_span(ids, value_ids)
                if span is None:
                    continue
                examples.append((hidden[0, : len(ids)].detach(), pointer.arg_idx[name], span[0], span[1]))
    if not examples:
        raise ValueError("stateful pointer training found no literal string spans")
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=2e-2)
    rng = random.Random(seed)
    pointer.train()
    for _ in range(max(1, steps)):
        batch = [examples[rng.randrange(len(examples))] for _ in range(min(16, len(examples)))]
        loss = torch.zeros((), device=device)
        for hidden, arg_index, start, end in batch:
            arg = torch.tensor([arg_index], dtype=torch.long, device=device)
            start_logits, end_logits = pointer.logits(hidden.unsqueeze(0), arg)
            loss = loss + F.cross_entropy(start_logits, torch.tensor([start], device=device))
            loss = loss + F.cross_entropy(end_logits, torch.tensor([end], device=device))
        optimizer.zero_grad(set_to_none=True)
        (loss / len(batch)).backward()
        optimizer.step()
    pointer.eval()


def _train_heads(
    model: LocalAgentLM,
    tokenizer,
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    tools,
    *,
    steps: int,
    seed: int,
    device: str,
    warm_start: bool,
) -> tuple[RouteHead, DenseToolSelector, PointerHead, dict[str, Any]]:
    torch.manual_seed(seed)
    model.eval()
    route = RouteHead(model.cfg.d_model).to(device)
    if warm_start and parent.get("route_head"):
        route.load_state_dict(parent["route_head"])
    feature_rows = _action_augmented_rows(rows)
    examples = _examples(feature_rows)
    dense = DenseToolSelector(model.cfg.d_model, proj=int(parent.get("selector_proj", 256))).to(device)
    if warm_start and parent.get("dense_selector"):
        dense.load_state_dict(parent["dense_selector"])
    embeddings = tool_embeddings(tools, device=device, examples=examples)
    feature_rows_tensor = _features(model, tokenizer, feature_rows, device)
    route_labels = torch.tensor(
        [ROUTE_INDEX[route_of(row["sample"].ref_name)] for row in feature_rows],
        dtype=torch.long,
        device=device,
    )
    selector_rows = [row for row in feature_rows if row["sample"].kind == "tool"]
    selector_features = _features(model, tokenizer, selector_rows, device)
    selector_labels = torch.tensor(
        [{tool.name: index for index, tool in enumerate(tools)}[row["sample"].ref_name] for row in selector_rows],
        dtype=torch.long,
        device=device,
    )
    route_optimizer = torch.optim.AdamW(route.parameters(), lr=5e-3)
    dense_optimizer = torch.optim.AdamW(dense.parameters(), lr=5e-3)
    rng = random.Random(seed)
    route.train()
    dense.train()
    for _ in range(max(1, steps)):
        route_idx = torch.tensor(
            [rng.randrange(len(feature_rows)) for _ in range(min(32, len(feature_rows)))], device=device
        )
        route_loss = F.cross_entropy(route(feature_rows_tensor[route_idx]), route_labels[route_idx])
        route_optimizer.zero_grad(set_to_none=True)
        route_loss.backward()
        route_optimizer.step()
        selector_idx = torch.tensor(
            [rng.randrange(len(selector_rows)) for _ in range(min(32, len(selector_rows)))],
            device=device,
        )
        selector_loss = F.cross_entropy(
            dense(selector_features[selector_idx], embeddings), selector_labels[selector_idx]
        )
        dense_optimizer.zero_grad(set_to_none=True)
        selector_loss.backward()
        dense_optimizer.step()
    route.eval()
    dense.eval()
    pointer = _warm_pointer(parent, model.cfg.d_model, random_init=not warm_start).to(device)
    _train_pointer(model, tokenizer, rows, pointer, steps=max(1, steps // 2), seed=seed, device=device)
    return route, dense, pointer, {
        "examples": examples,
        "route_features": len(feature_rows),
        "clean_route_features": len(rows),
        "selector_features": len(selector_rows),
        "pointer_span_examples": sum(
            isinstance(value, str)
            for row in rows
            if row["sample"].kind == "tool"
            for value in json.loads(row["sample"].ref_args).values()
        ),
    }


def _selector_metrics(model, tokenizer, rows, route, dense, tools, examples, device: str) -> dict[str, Any]:
    bound = BoundSelector(dense, tools, device=device, examples=examples)
    route_correct = 0
    top1 = 0
    top3 = 0
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "route": 0, "top1": 0, "top3": 0})
    for row in rows:
        sample = row["sample"]
        feature = _feat(model, tokenizer, sample.prompt, device)
        family = row["task"].family
        stats = by_family[family]
        stats["rows"] += 1
        expected_route = route_of(sample.ref_name)
        route_ok = ROUTES[int(route(feature).argmax(-1))] == expected_route
        route_correct += int(route_ok)
        stats["route"] += int(route_ok)
        if sample.kind != "tool":
            continue
        ranked = bound.rank(feature)
        top1_ok = ranked[0] == sample.ref_name
        top3_ok = sample.ref_name in ranked[:3]
        top1 += int(top1_ok)
        top3 += int(top3_ok)
        stats["top1"] += int(top1_ok)
        stats["top3"] += int(top3_ok)
    tool_rows = sum(row["sample"].kind == "tool" for row in rows)
    return {
        "rows": len(rows),
        "tool_rows": tool_rows,
        "route_accuracy": route_correct / max(1, len(rows)),
        "selector_top1": top1 / max(1, tool_rows),
        "selector_top3": top3 / max(1, tool_rows),
        "by_family": {
            family: {
                **stats,
                "route_accuracy": stats["route"] / max(1, stats["rows"]),
                "selector_top1": stats["top1"] / max(1, stats["rows"] - int(family == "abstention")),
                "selector_top3": stats["top3"] / max(1, stats["rows"] - int(family == "abstention")),
            }
            for family, stats in sorted(by_family.items())
        },
    }


@torch.no_grad()
def _closed_loop_metrics(model, tokenizer, tasks, route, dense, pointer, tools, examples, device: str) -> dict[str, Any]:
    bound = BoundSelector(dense, tools, device=device, examples=examples)
    totals = Counter()
    shaped_reward_total = 0.0
    families: dict[str, Counter] = defaultdict(Counter)
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        state = copy.deepcopy(task.initial_state)
        task_counts = Counter()
        for index, expected in enumerate(task.actions):
            prompt = state_prompt(task, index, state)
            output = hybrid_decode(
                model,
                tokenizer,
                prompt,
                tools,
                device=device,
                selector=bound,
                route_head=route,
                ptr_head=pointer,
                top_m=1,
            )
            calls = extract_tool_calls(output)
            predicted = calls[0] if calls else None
            result = apply_action(
                task,
                index,
                state,
                predicted.name if predicted else None,
                predicted.arguments if predicted else {},
            )
            state = result.state
            shaped_reward_total += stateful_reward(result)
            metrics = {
                "schema_valid": result.schema_valid,
                "exact_tool": result.exact_tool,
                "exact_args": result.exact_args,
                "exact_action": result.exact_action,
                "state_transition": result.state_transition,
                "closed_loop_success": result.closed_loop_success,
            }
            for key, value in metrics.items():
                totals[key] += int(value)
                task_counts[key] += int(value)
                families[task.family][key] += int(value)
            task_counts["steps"] += 1
        complete = task_complete(task, state)
        if complete:
            # Add the terminal bonus once per completed task; intermediate rewards remain per-step.
            shaped_reward_total += stateful_reward_spec()["terminal"]
        totals["task_complete"] += int(complete)
        families[task.family]["task_complete"] += int(complete)
        task_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "steps": task_counts["steps"],
                "closed_loop_success": task_counts["closed_loop_success"],
                "task_complete": complete,
            }
        )
    total_steps = sum(row["steps"] for row in task_rows)
    return {
        "steps": total_steps,
        "mean_shaped_reward": shaped_reward_total / max(1, total_steps),
        "reward_spec": stateful_reward_spec(),
        "schema_valid_rate": totals["schema_valid"] / max(1, total_steps),
        "exact_tool_rate": totals["exact_tool"] / max(1, total_steps),
        "exact_args_rate": totals["exact_args"] / max(1, total_steps),
        "exact_action_rate": totals["exact_action"] / max(1, total_steps),
        "state_transition_rate": totals["state_transition"] / max(1, total_steps),
        "closed_loop_success_rate": totals["closed_loop_success"] / max(1, total_steps),
        "task_complete_rate": totals["task_complete"] / max(1, len(tasks)),
        "recovery_task_complete_rate": sum(
            row["task_complete"] for row in task_rows if row["family"] == "recovery"
        ) / max(1, sum(row["family"] == "recovery" for row in task_rows)),
        "abstention_exact": sum(
            row["task_complete"] for row in task_rows if row["family"] == "abstention"
        ) / max(1, sum(row["family"] == "abstention" for row in task_rows)),
        "by_family": {
            family: {
                "steps": sum(row["steps"] for row in task_rows if row["family"] == family),
                "closed_loop_success_rate": stats["closed_loop_success"] / max(
                    1, sum(row["steps"] for row in task_rows if row["family"] == family)
                ),
                "task_complete": stats["task_complete"],
            }
            for family, stats in sorted(families.items())
        },
        "task_rows": task_rows,
    }


def _arm(
    parent: dict[str, Any],
    parent_model: LocalAgentLM,
    tokenizer,
    train_rows,
    eval_tasks,
    tools,
    *,
    label: str,
    seed: int,
    steps: int,
    device: str,
    random_backbone: bool,
    warm_start: bool,
) -> dict[str, Any]:
    cfg = ModelConfig(**parent["cfg"])
    if random_backbone:
        torch.manual_seed(seed)
        model = LocalAgentLM(cfg)
    else:
        model = copy.deepcopy(parent_model)
    model.to(device).eval()
    route, dense, pointer, train_info = _train_heads(
        model,
        tokenizer,
        parent,
        train_rows,
        tools,
        steps=steps,
        seed=seed,
        device=device,
        warm_start=warm_start,
    )
    examples = train_info["examples"]
    eval_rows = _rows(eval_tasks)
    selector = _selector_metrics(model, tokenizer, eval_rows, route, dense, tools, examples, device)
    closed_loop = _closed_loop_metrics(
        model, tokenizer, eval_tasks, route, dense, pointer, tools, examples, device
    )
    public_training = {key: value for key, value in train_info.items() if key != "examples"}
    public_training["example_counts"] = {
        name: len(values) for name, values in sorted(examples.items())
    }
    public_training["example_prompt_sha256"] = {
        name: _state_hash(sorted(values)) for name, values in sorted(examples.items())
    }
    return {
        "label": label,
        "random_backbone": random_backbone,
        "warm_start_heads": warm_start,
        "training": public_training | {"steps": steps, "seed": seed},
        "selector": selector,
        "closed_loop": closed_loop,
        "head_hashes": {
            "route": _tensor_hash(route.state_dict()),
            "dense": _tensor_hash(dense.state_dict()),
            "pointer": _tensor_hash(pointer.state_dict()),
        },
        "heads": {"route": route, "dense": dense, "pointer": pointer, "examples": examples},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite stateful probe outputs")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    parent, parent_model, tokenizer = _load_parent(args.init)
    train_tasks = build_tasks("train")
    eval_tasks = build_tasks("eval")
    train_rows = _rows(train_tasks)
    tools = tool_specs()
    tool_names = {tool.name for tool in tools}
    train_names = {row["sample"].ref_name for row in train_rows if row["sample"].kind == "tool"}
    if not train_names <= tool_names:
        raise ValueError(f"stateful training references unknown tools: {sorted(train_names - tool_names)}")
    transfer = _arm(
        parent,
        parent_model,
        tokenizer,
        train_rows,
        eval_tasks,
        tools,
        label="pretrained_frozen_backbone",
        seed=2029,
        steps=args.steps,
        device=args.device,
        random_backbone=False,
        warm_start=True,
    )
    random_control = _arm(
        parent,
        parent_model,
        tokenizer,
        train_rows,
        eval_tasks,
        tools,
        label="matched_random_backbone",
        seed=2029,
        steps=args.steps,
        device=args.device,
        random_backbone=True,
        warm_start=False,
    )
    transfer_metrics = {key: value for key, value in transfer.items() if key != "heads"}
    random_metrics = {key: value for key, value in random_control.items() if key != "heads"}
    child = dict(parent)
    child.update(
        {
            "stage": "stateful_productivity_probe",
            "stateful_pointer_args": STATEFUL_PTR_ARGS,
            "route_head": transfer["heads"]["route"].state_dict(),
            "dense_selector": transfer["heads"]["dense"].state_dict(),
            "ptr_head": transfer["heads"]["pointer"].state_dict(),
            "ptr_args": STATEFUL_PTR_ARGS,
            "examples": transfer["heads"]["examples"],
            "stateful_probe": transfer_metrics,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    output_identity = {"path": str(args.output), "bytes": _sha256(args.output)[0], "sha256": _sha256(args.output)[1]}
    report = {
        "kind": "localagent_stateful_productivity_transfer_probe",
        "schema_version": 1,
        "suite": SUITE_ID,
        "source": {
            "kind": "local_synthetic_state_machine",
            "train_inventory": suite_inventory("train"),
            "eval_inventory": suite_inventory("eval"),
            "train_task_hash": _state_hash([task.task_id for task in train_tasks]),
            "eval_task_hash": _state_hash([task.task_id for task in eval_tasks]),
            "public_benchmark_text_used": False,
            "external_accounts_used": False,
            "tools_executed": False,
            "native_runtime_executed": False,
        },
        "parent": {"path": str(args.init), "bytes": _sha256(args.init)[0], "sha256": _sha256(args.init)[1]},
        "configuration": {
            "model_config": parent["cfg"],
            "tool_pool_size": len(tools),
            "tool_pool_sha256": _state_hash([tool.name for tool in tools]),
            "steps": args.steps,
            "seed": 2029,
            "device": args.device,
            "stateful_pointer_args": STATEFUL_PTR_ARGS,
            "reward_spec": stateful_reward_spec(),
        },
        "arms": {
            "pretrained_frozen_backbone": transfer_metrics,
            "matched_random_backbone": random_metrics,
        },
        "comparison": {
            "selector_top1_random_minus_pretrained": random_control["selector"]["selector_top1"] - transfer["selector"]["selector_top1"],
            "selector_top3_random_minus_pretrained": random_control["selector"]["selector_top3"] - transfer["selector"]["selector_top3"],
            "closed_loop_random_minus_pretrained": random_control["closed_loop"]["closed_loop_success_rate"] - transfer["closed_loop"]["closed_loop_success_rate"],
            "task_complete_random_minus_pretrained": random_control["closed_loop"]["task_complete_rate"] - transfer["closed_loop"]["task_complete_rate"],
        },
        "output": output_identity,
        "claim_boundary": (
            "This is a local state-machine transfer probe. It reports selector, pointer-copy, "
            "schema, state-transition, recovery, abstention, and closed-loop diagnostics only. "
            "It is not a public benchmark score, emulator/browser/MCP execution, real email or "
            "Notion operation, or evidence of native WebGPU throughput."
        ),
    }
    report["receipt_self_sha256"] = _state_hash(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
