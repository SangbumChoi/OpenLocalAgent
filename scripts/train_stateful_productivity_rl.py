#!/usr/bin/env python
"""Run a small, verifiable GRPO continuation on the local productivity state machine.

The public benchmark adapters in this repository are intentionally kept separate from training.
This command supplies the missing RL-simulation stage for the requested email/Notion/browser
surface: it serializes the canonical train/eval ``Conversation`` rows, runs the existing pure
PyTorch GRPO implementation, and writes a hash-bound receipt.  The environment is represented by
deterministic tool responses from ``stateful_productivity``; no public benchmark text, emulator,
browser, MCP service, or external account is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from localagent.data.schema import Conversation, Message, Role, ToolCall
from localagent.data.agent_synth import Sample
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.data.stateful_productivity import (
    SUITE_ID,
    apply_action,
    build_tasks,
    state_prompt,
    tool_specs,
)
from localagent.train.rl import run as run_rl
from localagent.train.sft import sft
from localagent.train.stage_data import canonical_sha256


_DEPLOYMENT_HEADS = (
    "tool_head",
    "ptr_head",
    "route_head",
    "dense_selector",
    "selector_proj",
    "value_head",
)


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_rows(path: Path, split: str) -> dict[str, Any]:
    tasks = build_tasks(split)
    catalog = {tool.name: tool for tool in tool_specs()}
    rows = []
    for task in tasks:
        # Full 62-tool catalogs exceed the 2K context of the WebGPU tier.  The native methods we
        # are rehearsing all use task-scoped candidate tools, so bind only the tools that appear
        # in this workflow (plus no tool for abstention).
        names = {action.tool for action in task.actions if action.tool is not None}
        task_tools = [catalog[name] for name in sorted(names)]
        state = copy.deepcopy(task.initial_state)
        for index, action in enumerate(task.actions):
            prompt = state_prompt(task, index, state)
            if split == "eval" and task.family == "abstention":
                # The abstention task is intentionally slot-free and otherwise identical across
                # splits.  Keep the strict legacy semantic-overlap audit leakage-safe.
                prompt += " (evaluation split)"
            if action.tool is None:
                assistant = Message(role=Role.assistant, content="I won't invoke a tool.")
            else:
                assistant = Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name=action.tool, arguments=copy.deepcopy(action.arguments))],
                )
            rows.append(
                Conversation(
                    messages=[Message(role=Role.user, content=prompt), assistant],
                    tools=task_tools,
                    meta={
                        "kind": "stateful_productivity_rl_decision",
                        "suite": SUITE_ID,
                        "task_id": task.task_id,
                        "family": task.family,
                        "split": split,
                        "step": index,
                    },
                )
            )
            if action.tool is not None:
                state = apply_action(task, index, state, action.tool, action.arguments).state
    path.write_text("".join(row.to_json() + "\n" for row in rows), encoding="utf-8")
    size, digest = _sha256(path)
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "tasks": [task.task_id for task in tasks],
        "task_hash": canonical_sha256([task.task_id for task in tasks]),
    }


def _write_config(
    path: Path,
    *,
    parent: Path,
    train_path: Path,
    eval_path: Path,
    out_dir: Path,
    steps: int,
    prompts_per_step: int,
    group_size: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "stage": "rl",
        "model_config": "configs/model/webgpu-10m-hybrid.yaml",
        "init_from": str(parent),
        "data": {
            "strict_conversation_artifacts": False,
            "conversations": [str(train_path)],
            "eval_conversations": [str(eval_path)],
            "tokenizer": {"kind": "bpe", "path": "data/tokenizer-webgpu-proxy-16k.json"},
        },
        "environment": {"name": "stateful_productivity", "learned_judge": False},
        "rollout": {
            "prompts_per_step": prompts_per_step,
            "group_size": group_size,
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,
        },
        "policy": {"clip_ratio": 0.2, "kl_beta": 0.02, "epochs_per_rollout": 1},
        "reward": {"format_weight": 0.1, "truncation_penalty": 0.05},
        "optim": {"name": "adamw", "lr": 2.0e-5},
        "schedule": {"total_steps": steps, "warmup_steps": 1},
        "runtime": {"device": "cpu", "dtype": "float32", "seed": seed},
        "log": {"out_dir": str(out_dir), "ckpt_every": max(1, steps)},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def _sft_samples(split: str) -> list[Sample]:
    """Build single-turn state-conditioned SFT rows for the warm-start prelude."""

    rows: list[Sample] = []
    for task in build_tasks(split):
        state = copy.deepcopy(task.initial_state)
        for index, action in enumerate(task.actions):
            prompt = state_prompt(task, index, state)
            if action.tool is None:
                rows.append(
                    Sample(
                        "stateful_productivity_rl",
                        task.family,
                        prompt,
                        "text",
                        "I won't invoke a tool.",
                        "text",
                        "{}",
                    )
                )
            else:
                rows.append(
                    Sample(
                        "stateful_productivity_rl",
                        task.family,
                        prompt,
                        "tool",
                        json.dumps(
                            {"arguments": action.arguments, "name": action.tool},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        action.tool,
                        json.dumps(action.arguments, sort_keys=True, separators=(",", ":")),
                    )
                )
                state = apply_action(task, index, state, action.tool, action.arguments).state
    return rows


def _warm_sft_parent(
    parent: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Run a bounded stateful SFT prelude and save an RL-compatible ``stage=sft`` parent."""

    import torch

    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["cfg"])
    model = LocalAgentLM(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer_meta = checkpoint.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(tokenizer_meta.get("kind", "byte"), tokenizer_meta.get("path"))
    losses, _, _, metrics = sft(
        model,
        _sft_samples("train"),
        tokenizer,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        warmup=max(1, min(8, steps // 4)),
        device="cpu",
        seed=seed,
        log=lambda *_args: None,
        return_metrics=True,
    )
    child = dict(checkpoint)
    child.update(
        {
            "state_dict": model.state_dict(),
            "stage": "sft",
            "step": int(checkpoint.get("step", 0)) + steps,
            "loss_history": losses,
            "stateful_productivity_sft": {
                "steps": steps,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "metrics": metrics,
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, output)
    return {"steps": steps, "batch_size": batch_size, "lr": lr, "seed": seed, "metrics": metrics}


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    heldout = metrics.get("heldout_eval", {})
    pre = heldout.get("pre", {})
    post = heldout.get("post", {})
    return {
        "mean_reward_last": metrics.get("mean_reward_last"),
        "reward_steps": metrics.get("reward_steps"),
        "exact_match_accuracy_pre": pre.get("exact_match_accuracy"),
        "exact_match_accuracy_post": post.get("exact_match_accuracy"),
        "tool_exact_match_accuracy_pre": pre.get("tool_exact_match_accuracy"),
        "tool_exact_match_accuracy_post": post.get("tool_exact_match_accuracy"),
        "mean_reward_pre": pre.get("mean_reward"),
        "mean_reward_post": post.get("mean_reward"),
        "rl_accounting": metrics.get("rl_accounting"),
    }


def _preserve_deployment_heads(parent: Path, child: Path) -> list[str]:
    """Carry frozen action heads through an LM-only RL checkpoint.

    The GRPO loop optimizes the language-model state dict, while deployment uses auxiliary
    tool/route/pointer/selector heads stored beside it.  ``run_rl`` intentionally writes only the
    optimized model state, so copy compatible parent heads back before the receipt is assembled.
    This keeps the child loadable by the WebGPU agent and makes the heads' frozen status explicit.
    """

    import torch

    parent_checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    child_checkpoint = torch.load(child, map_location="cpu", weights_only=False)
    if not isinstance(parent_checkpoint, dict) or not isinstance(child_checkpoint, dict):
        raise TypeError("RL checkpoint and parent checkpoint must be mappings")
    preserved: list[str] = []
    for name in _DEPLOYMENT_HEADS:
        value = parent_checkpoint.get(name)
        if value is None or name in child_checkpoint:
            continue
        child_checkpoint[name] = copy.deepcopy(value)
        preserved.append(name)
    # ``ptr_head`` is coupled to its argument vocabulary.  RL writes only the language-model
    # state dict, so preserve the metadata that lets deployment reconstruct the same pointer
    # shape instead of silently falling back to the legacy 17-argument vocabulary.
    if "ptr_args" not in child_checkpoint and "ptr_args" in parent_checkpoint:
        child_checkpoint["ptr_args"] = copy.deepcopy(parent_checkpoint["ptr_args"])
    child_checkpoint["stateful_productivity_rl"] = {
        "deployment_heads": preserved,
        "deployment_heads_trainable": False,
        "deployment_heads_source": str(parent),
    }
    torch.save(child_checkpoint, child)
    return preserved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-lr", type=float, default=1.0e-4)
    args = parser.parse_args()
    if args.work_dir.exists():
        raise SystemExit(f"refusing to overwrite work directory {args.work_dir}")
    if args.receipt.exists():
        raise SystemExit(f"refusing to overwrite receipt {args.receipt}")
    for value, label in (
        (args.steps, "steps"),
        (args.prompts_per_step, "prompts-per-step"),
        (args.group_size, "group-size"),
        (args.max_new_tokens, "max-new-tokens"),
        (args.sft_steps, "sft-steps"),
        (args.sft_batch_size, "sft-batch-size"),
    ):
        if value < 1:
            raise ValueError(f"{label} must be positive")
    if not args.parent.is_file():
        raise FileNotFoundError(args.parent)

    data_dir = args.work_dir / "data"
    out_dir = args.work_dir / "rl"
    data_dir.mkdir(parents=True)
    train_identity = _write_rows(data_dir / "stateful-productivity-train.jsonl", "train")
    eval_identity = _write_rows(data_dir / "stateful-productivity-eval.jsonl", "eval")
    sft_parent = args.work_dir / "sft-parent.pt"
    sft_training = _warm_sft_parent(
        args.parent,
        sft_parent,
        steps=args.sft_steps,
        batch_size=args.sft_batch_size,
        lr=args.sft_lr,
        seed=args.seed,
    )
    config_path = args.work_dir / "rl.yaml"
    config = _write_config(
        config_path,
        parent=sft_parent,
        train_path=Path(train_identity["path"]),
        eval_path=Path(eval_identity["path"]),
        out_dir=out_dir,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    run_rl(str(config_path))
    metrics_path = out_dir / "metrics.json"
    checkpoint_path = out_dir / "latest.pt"
    preserved_heads = _preserve_deployment_heads(sft_parent, checkpoint_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    original_parent_bytes, original_parent_sha = _sha256(args.parent)
    parent_bytes, parent_sha = _sha256(sft_parent)
    checkpoint_bytes, checkpoint_sha = _sha256(checkpoint_path)
    config_bytes, config_sha = _sha256(config_path)
    report = {
        "kind": "localagent_stateful_productivity_grpo_simulation",
        "schema_version": 1,
        "suite": SUITE_ID,
        "source": {
            "kind": "local_deterministic_state_machine",
            "public_benchmark_text_used": False,
            "native_runtime_executed": False,
            "external_accounts_used": False,
            "tool_side_effects": "in_memory_only",
            "train": train_identity,
            "eval": eval_identity,
        },
        "parent": {
            "original": {"path": str(args.parent), "bytes": original_parent_bytes, "sha256": original_parent_sha},
            "sft": {"path": str(sft_parent), "bytes": parent_bytes, "sha256": parent_sha},
        },
        "configuration": {
            "config_sha256": config_sha,
            "config_bytes": config_bytes,
            "steps": args.steps,
            "prompts_per_step": args.prompts_per_step,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "sft": sft_training,
            "model_config": config["model_config"],
            "reward_environment": config["environment"]["name"],
            "deployment_heads_preserved": preserved_heads,
            "deployment_heads_trainable": False,
        },
        "training": _summary(metrics),
        "output": {
            "metrics": {"path": str(metrics_path), **dict(zip(("bytes", "sha256"), _sha256(metrics_path)))},
            "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_bytes, "sha256": checkpoint_sha},
        },
        "claim_boundary": (
            "This is an actual pure-PyTorch GRPO update over canonical local stateful email, "
            "Notion, browser, recovery, and abstention conversations. It is an RL simulation: "
            "the reward is opt-in schema/tool/argument/expected-transition shaping and the "
            "serialized tool responses are local fixtures. It is not public benchmark training, native Android/browser/MCP "
            "execution, real-account side effects, or evidence of WebGPU throughput."
        ),
    }
    report["receipt_self_sha256"] = canonical_sha256(report)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
