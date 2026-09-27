#!/usr/bin/env python
"""Compare frozen, low-rate-unfrozen, and random transfer on the local stateful probe.

This is the causal companion to ``train_stateful_productivity_probe.py``.  All three arms use the
same synthetic train/eval tasks, seed, tool pool, head budgets, and closed-loop evaluator.  The
only treatment is whether the verified pretrained backbone is frozen, updated with a small
learning rate, or replaced by a shape-matched random initialization.  It is a transfer decision
probe, not an official AndroidWorld, BrowserGym, OSWorld, MCPMark, or real-account score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import train_stateful_productivity_probe as base
from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings
from localagent.agent.routes import ROUTE_INDEX, RouteHead, route_of
from localagent.data.stateful_productivity import (
    SUITE_ID,
    build_tasks,
    canonical_json,
    stateful_reward_spec,
    suite_inventory,
)
from localagent.model import LocalAgentLM
from localagent.model.tokenizer import ASSISTANT, USER


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


def _batch_features(
    model: LocalAgentLM,
    tokenizer,
    prompts: list[str],
    device: str,
) -> torch.Tensor:
    """Compute differentiable final prompt features with causal-safe right padding.

    Right-padding is safe for a causal decoder because future padding positions cannot affect the
    earlier content positions whose features are selected.  Keeping the function here (rather
    than using the frozen ``_feat`` helper) is what allows the low-rate arm to update the backbone.
    """

    encoded = [tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")[-model.cfg.max_seq_len :] for prompt in prompts]
    width = max(len(row) for row in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, row in enumerate(encoded):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    _, hidden = model(ids, return_hidden=True)
    return torch.stack([hidden[index, len(row) - 1] for index, row in enumerate(encoded)])


def _movement_groups(parent_state: dict[str, torch.Tensor], child_state: dict[str, torch.Tensor]) -> dict[str, float]:
    groups = {
        "embedding": ("embed.", "in_proj.", "out_proj."),
        "mixer": ("blocks.",),
        "ffn": ("blocks.",),
        "normalization": ("norm.",),
    }
    values: dict[str, float] = {}
    values["backbone"] = base._relative_l2(parent_state, child_state)
    for group, prefixes in groups.items():
        names = []
        for name in parent_state:
            if group == "mixer" and ".attn." not in name:
                continue
            if group == "ffn" and ".ffn." not in name:
                continue
            if group == "normalization" and not ("norm." in name or name == "norm.weight"):
                continue
            if group == "embedding" and not name.startswith(prefixes):
                continue
            if group in {"mixer", "ffn", "normalization"} and not name.startswith("blocks.") and group != "normalization":
                continue
            if name in child_state:
                names.append(name)
        values[group] = base._relative_l2(
            {name: parent_state[name] for name in names},
            {name: child_state[name] for name in names},
        )
    return values


def _train_lowrate(
    parent: dict[str, Any],
    parent_model: LocalAgentLM,
    tokenizer,
    train_rows: list[dict[str, Any]],
    eval_tasks,
    tools,
    *,
    steps: int,
    seed: int,
    device: str,
    backbone_lr: float,
    head_lr: float,
    batch_size: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = copy.deepcopy(parent_model).to(device)
    route = RouteHead(model.cfg.d_model).to(device)
    if parent.get("route_head"):
        route.load_state_dict(parent["route_head"])
    # A deployed runtime appends a compact error observation after a rejected call and retries
    # against the same state.  The original probe only trained on clean state transitions, so a
    # policy could achieve good teacher-forced selection while repeating its failed action in the
    # actual loop.  Add one deterministic action-mismatch view per decision; the gold action and
    # state remain unchanged, and eval rows stay untouched.
    error_rows = []
    for row in train_rows:
        sample = row["sample"]
        error_rows.append(
            {
                **row,
                "sample": replace(
                    sample,
                    prompt=f"{sample.prompt} Last tool result: error=action_mismatch",
                ),
            }
        )
    feature_rows = base._action_augmented_rows([*train_rows, *error_rows])
    examples = base._examples(feature_rows)
    dense = DenseToolSelector(model.cfg.d_model, proj=int(parent.get("selector_proj", 256))).to(device)
    if parent.get("dense_selector"):
        dense.load_state_dict(parent["dense_selector"])
    embeddings = tool_embeddings(tools, device=device, examples=examples)
    name_index = {tool.name: index for index, tool in enumerate(tools)}
    selector_rows = [row for row in feature_rows if row["sample"].kind == "tool"]
    route_labels = [ROUTE_INDEX[route_of(row["sample"].ref_name)] for row in feature_rows]
    selector_labels = [name_index[row["sample"].ref_name] for row in selector_rows]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": backbone_lr},
            {"params": route.parameters(), "lr": head_lr},
            {"params": dense.parameters(), "lr": head_lr},
        ]
    )
    rng = random.Random(seed)
    model.train()
    route.train()
    dense.train()
    for _ in range(max(1, steps)):
        route_index = [rng.randrange(len(feature_rows)) for _ in range(min(batch_size, len(feature_rows)))]
        selector_index = [rng.randrange(len(selector_rows)) for _ in range(min(batch_size, len(selector_rows)))]
        route_prompts = [feature_rows[index]["sample"].prompt for index in route_index]
        selector_prompts = [selector_rows[index]["sample"].prompt for index in selector_index]
        features = _batch_features(model, tokenizer, route_prompts + selector_prompts, device)
        route_features = features[: len(route_prompts)]
        selector_features = features[len(route_prompts) :]
        route_target = torch.tensor([route_labels[index] for index in route_index], device=device)
        selector_target = torch.tensor([selector_labels[index] for index in selector_index], device=device)
        loss = F.cross_entropy(route(route_features), route_target) + F.cross_entropy(
            dense(selector_features, embeddings), selector_target
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    route.eval()
    dense.eval()
    pointer = base._warm_pointer(parent, model.cfg.d_model, random_init=False).to(device)
    base._train_pointer(
        model,
        tokenizer,
        train_rows,
        pointer,
        steps=max(1, steps // 2),
        seed=seed,
        device=device,
    )
    eval_rows = base._rows(eval_tasks)
    selector = base._selector_metrics(model, tokenizer, eval_rows, route, dense, tools, examples, device)
    closed_loop = base._closed_loop_metrics(
        model, tokenizer, eval_tasks, route, dense, pointer, tools, examples, device
    )
    train_info = {
        "route_features": len(feature_rows),
        "clean_route_features": len(train_rows),
        "error_recovery_features": len(error_rows),
        "selector_features": len(selector_rows),
        "pointer_span_examples": sum(
            isinstance(value, str)
            for row in train_rows
            if row["sample"].kind == "tool"
            for value in json.loads(row["sample"].ref_args).values()
        ),
        "example_counts": {name: len(values) for name, values in sorted(examples.items())},
        "example_prompt_sha256": {
            name: _state_hash(sorted(values)) for name, values in sorted(examples.items())
        },
    }
    return {
        "model": model,
        "route": route,
        "dense": dense,
        "pointer": pointer,
        "examples": examples,
        "selector": selector,
        "closed_loop": closed_loop,
        "training": train_info | {"steps": steps, "seed": seed, "backbone_lr": backbone_lr, "head_lr": head_lr, "batch_size": batch_size},
        "weight_movement": _movement_groups(parent["state_dict"], model.state_dict()),
    }


def _public_arm(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"model", "route", "dense", "pointer", "examples", "heads"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite low-rate probe outputs")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")

    parent, parent_model, tokenizer = base._load_parent(args.init)
    train_tasks = build_tasks("train")
    eval_tasks = build_tasks("eval")
    train_rows = base._rows(train_tasks)
    tools = base.tool_specs()
    frozen = base._arm(
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
    random_control = base._arm(
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
    lowrate = _train_lowrate(
        parent,
        parent_model,
        tokenizer,
        train_rows,
        eval_tasks,
        tools,
        steps=args.steps,
        seed=2029,
        device=args.device,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
    )
    child = dict(parent)
    child.update(
        {
            "stage": "stateful_productivity_lowrate_probe",
            "state_dict": lowrate["model"].state_dict(),
            "route_head": lowrate["route"].state_dict(),
            "dense_selector": lowrate["dense"].state_dict(),
            "ptr_head": lowrate["pointer"].state_dict(),
            "ptr_args": base.STATEFUL_PTR_ARGS,
            "examples": lowrate["examples"],
            "stateful_probe": _public_arm(lowrate),
            "stateful_probe_transfer": {
                "backbone_lr": args.backbone_lr,
                "head_lr": args.head_lr,
                "backbone_movement": lowrate["weight_movement"],
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    size, digest = _sha256(args.output)
    arms = {
        "pretrained_frozen_backbone": _public_arm(frozen),
        "pretrained_lowrate_unfrozen_backbone": _public_arm(lowrate),
        "matched_random_backbone": _public_arm(random_control),
    }
    frozen_closed = frozen["closed_loop"]["closed_loop_success_rate"]
    lowrate_closed = lowrate["closed_loop"]["closed_loop_success_rate"]
    random_closed = random_control["closed_loop"]["closed_loop_success_rate"]
    report = {
        "kind": "localagent_stateful_productivity_transfer_ablation",
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
            "batch_size": args.batch_size,
            "seed": 2029,
            "device": args.device,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "reward_spec": stateful_reward_spec(),
        },
        "arms": arms,
        "comparison": {
            "lowrate_minus_frozen_selector_top1": lowrate["selector"]["selector_top1"] - frozen["selector"]["selector_top1"],
            "lowrate_minus_random_selector_top1": lowrate["selector"]["selector_top1"] - random_control["selector"]["selector_top1"],
            "lowrate_minus_frozen_closed_loop": lowrate_closed - frozen_closed,
            "lowrate_minus_random_closed_loop": lowrate_closed - random_closed,
            "transfer_adoption_decision": (
                "retain_pretrained_backbone_for_next_native_probe"
                if lowrate_closed > random_closed and lowrate["selector"]["selector_top1"] > random_control["selector"]["selector_top1"]
                else "do_not_adopt_as_capability_evidence"
            ),
        },
        "output": {"path": str(args.output), "bytes": size, "sha256": digest},
        "claim_boundary": (
            "This is a matched local state-machine transfer ablation. Weight movement and held-out "
            "selector/closed-loop diagnostics do not establish native mobile, browser, desktop, "
            "MCP, real email, Notion, or WebGPU capability."
        ),
    }
    report["receipt_self_sha256"] = _state_hash(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
