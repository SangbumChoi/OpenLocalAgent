#!/usr/bin/env python3
"""Train a catalog-aware dense selector on the ToolACE action-history projection.

The action-history SFT child keeps the language-model backbone warm but does not update the
dispatch heads.  This probe adapts only the two-tower selector, using the exact catalog + history
context consumed by the WebGPU free-run evaluator.  It reports an inherited-selector arm and a
matched random-backbone arm, then writes the transferred selector into a child checkpoint.

This is a public-source, text-first diagnostic.  It does not execute ToolACE, a browser, an
emulator, MCP server, email, or Notion side effects, and it is not an official ToolACE/BFCL score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from localagent.agent.dense_selector import (
    DenseToolSelector,
    tool_embeddings,
)
from localagent.agent.tool_head import _feat
from localagent.data.prompt_contract import render_function_catalog
from localagent.data.render import history_text
from localagent.data.schema import Conversation, Role, ToolSpec
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, BPE_EOS, load_tokenizer

SOURCE_URL = "https://huggingface.co/datasets/Team-ACE/ToolACE"
SOURCE_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
SEED = 2031


@dataclass(frozen=True)
class SelectorDecision:
    """One assistant action with its row-local candidate catalog."""

    prompt: str
    target: str
    tools: tuple[ToolSpec, ...]


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> list[Conversation]:
    rows = [Conversation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty ToolACE input: {path}")
    return rows


def _catalog(rows: list[Conversation]) -> list[ToolSpec]:
    by_name: dict[str, ToolSpec] = {}
    for row in rows:
        for tool in row.tools:
            by_name.setdefault(tool.name, tool)
    return [by_name[name] for name in sorted(by_name)]


def _decisions(rows: list[Conversation]) -> list[SelectorDecision]:
    decisions: list[SelectorDecision] = []
    for row in rows:
        catalog = render_function_catalog(row.tools) + BPE_EOS
        for index, message in enumerate(row.messages):
            if message.role != Role.assistant or not message.tool_calls:
                continue
            history = history_text(row.messages[:index])
            decisions.append(
                SelectorDecision(
                    prompt=catalog + history + ASSISTANT,
                    target=message.tool_calls[0].name,
                    tools=tuple(row.tools),
                )
            )
    if not decisions:
        raise ValueError("ToolACE input contains no assistant tool actions")
    return decisions


def _selector_from_state(model: LocalAgentLM, checkpoint: dict[str, Any]) -> DenseToolSelector:
    state = checkpoint.get("dense_selector")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no dense_selector state")
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=int(state["t_proj.weight"].shape[1]),
        proj=int(state["q_proj.weight"].shape[0]),
    )
    selector.load_state_dict(state)
    return selector.eval()


def _score(
    model: LocalAgentLM,
    tokenizer: Any,
    selector: DenseToolSelector,
    decisions: list[SelectorDecision],
) -> dict[str, Any]:
    selector.eval()
    top1 = top3 = covered = 0
    candidate_counts: list[int] = []
    for decision in decisions:
        names = [tool.name for tool in decision.tools]
        if decision.target not in names:
            raise ValueError(f"target {decision.target!r} is absent from its catalog")
        feature = _feat(model, tokenizer, decision.prompt, "cpu", framed=True)
        embeddings = tool_embeddings(decision.tools, selector.t_proj.weight.shape[1])
        with torch.no_grad():
            ranking = selector(feature.unsqueeze(0), embeddings)[0].argsort(descending=True).tolist()
        ranked = [names[index] for index in ranking]
        covered += 1
        candidate_counts.append(len(names))
        top1 += int(ranked[0] == decision.target)
        top3 += int(decision.target in ranked[:3])
    count = len(decisions)
    return {
        "actions": count,
        "candidate_coverage": covered / count,
        "candidate_count_mean": sum(candidate_counts) / count,
        "selector_top1": top1 / count,
        "selector_top3": top3 / count,
    }


def _train(
    model: LocalAgentLM,
    tokenizer: Any,
    decisions: list[SelectorDecision],
    tools: list[ToolSpec],
    *,
    init: DenseToolSelector | None,
    steps: int,
    batch_size: int,
    lr: float,
    proj: int,
) -> DenseToolSelector:
    model.eval()
    with torch.no_grad():
        features = torch.stack(
            [_feat(model, tokenizer, decision.prompt, "cpu", framed=True) for decision in decisions]
        )
    name_to_index = {tool.name: index for index, tool in enumerate(tools)}
    labels = torch.tensor([name_to_index[d.target] for d in decisions], dtype=torch.long)
    embeddings = tool_embeddings(tools, dim=init.emb_dim if init is not None else 8192)
    selector = init if init is not None else DenseToolSelector(model.cfg.d_model, embeddings.shape[1], proj)
    selector.train()
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr)
    rng = random.Random(SEED)
    for step in range(steps):
        indices = torch.tensor([rng.randrange(len(decisions)) for _ in range(batch_size)])
        loss = F.cross_entropy(selector(features[indices], embeddings), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return selector.eval()


def _movement(before: DenseToolSelector, after: DenseToolSelector) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, names in {
        "query_tower": ("q_proj.weight", "q_proj.bias"),
        "tool_tower": ("t_proj.weight", "t_proj.bias"),
    }.items():
        base = torch.cat([before.state_dict()[name].float().flatten() for name in names])
        delta = torch.cat(
            [(after.state_dict()[name].float() - before.state_dict()[name].float()).flatten() for name in names]
        )
        result[group] = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite selector-probe outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and learning rate must be positive")

    train_rows = _load(args.train)
    eval_rows = _load(args.eval)
    train_decisions = _decisions(train_rows)
    eval_decisions = _decisions(eval_rows)
    train_tools = _catalog(train_rows)
    all_tools = _catalog(train_rows + eval_rows)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    model.eval()
    metadata = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(metadata["kind"], metadata.get("path"))
    inherited = _selector_from_state(model, parent)
    inherited_metrics = _score(model, tokenizer, inherited, eval_decisions)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    transferred = _train(
        model,
        tokenizer,
        train_decisions,
        train_tools,
        init=_selector_from_state(model, parent),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        proj=int(parent["selector_proj"]),
    )
    transferred_metrics = _score(model, tokenizer, transferred, eval_decisions)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    random_model = LocalAgentLM(config)
    random_model.eval()
    random_selector = _train(
        random_model,
        tokenizer,
        train_decisions,
        train_tools,
        init=None,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        proj=int(parent["selector_proj"]),
    )
    random_metrics = _score(random_model, tokenizer, random_selector, eval_decisions)

    child = dict(parent)
    child.update(
        {
            "dense_selector": transferred.state_dict(),
            "stage": "sft_toolace_action_history_selector_probe",
            "parent_checkpoint_sha256": _identity(args.init)["sha256"],
            "toolace_action_history_selector_training": {
                "dataset": "Team-ACE/ToolACE",
                "url": SOURCE_URL,
                "revision": SOURCE_REVISION,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_actions": len(train_decisions),
                "eval_actions": len(eval_decisions),
                "train_catalog_tools": len(train_tools),
                "eval_union_catalog_tools": len(all_tools),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": SEED,
                "backbone_frozen": True,
                "route_head": "inherited_unchanged",
                "pointer_head": "inherited_unchanged",
                "inherited_eval": inherited_metrics,
                "transferred_eval": transferred_metrics,
                "random_eval": random_metrics,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)

    report = {
        "kind": "localagent_toolace_action_history_selector_transfer_probe",
        "schema_version": 1,
        "source": {
            "dataset": "Team-ACE/ToolACE",
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_actions": len(train_decisions),
            "eval_actions": len(eval_decisions),
            "train_catalog_tools": len(train_tools),
            "eval_union_catalog_tools": len(all_tools),
        },
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": SEED,
            "backbone_frozen": True,
            "context": "catalog + BPE_EOS + action-history + ASSISTANT",
        },
        "arms": {
            "inherited_pretrained_selector": {
                "backbone": "m173_warm_frozen",
                "selector": "inherited",
                "metrics": inherited_metrics,
            },
            "retrained_pretrained_backbone": {
                "backbone": "m173_warm_frozen",
                "selector": "retrained_on_toolace_action_history",
                "metrics": transferred_metrics,
                "selector_relative_movement": _movement(inherited, transferred),
            },
            "retrained_matched_random_backbone": {
                "backbone": "matched_random_frozen",
                "selector": "retrained_on_toolace_action_history",
                "metrics": random_metrics,
            },
        },
        "decision": {
            "transfer_improves_over_inherited_top1": transferred_metrics["selector_top1"]
            > inherited_metrics["selector_top1"],
            "transfer_beats_random_top1": transferred_metrics["selector_top1"]
            > random_metrics["selector_top1"],
            "adoption": "adopt_selector_only_if_free_run_improves",
            "reason": (
                "Candidate-list ranking is a necessary dispatch diagnostic, but the WebGPU-shaped "
                "free-run receipt remains the adoption gate."
            ),
        },
        "claim_boundary": (
            "ToolACE source-projected catalog-aware selector diagnostic only; no native execution, "
            "official ToolACE/BFCL score, screenshot grounding, or external side effects."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
