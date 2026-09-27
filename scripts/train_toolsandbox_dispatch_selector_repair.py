#!/usr/bin/env python3
"""Train a matched ToolSandbox-specific dense selector and preserve the WebGPU backbone.

The public ToolSandbox projection supplies row-local candidate tool lists.  This script trains the
two-tower selector over those lists while keeping the language model, route head, pointer head,
and tokenizer fixed.  A warm selector starts from the checkpoint state; the random arm starts
from a matched seed.  The output checkpoints are suitable for the native ToolSandbox runner, but
the report remains a source-projected diagnostic rather than an official benchmark score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings
from localagent.agent.tool_head import _feat
from localagent.data.schema import Conversation, ToolSpec
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import ProbeDecision, probe_decisions

SOURCE_URL = "https://github.com/apple/ToolSandbox"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
SEED = 2046


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
        raise ValueError(f"empty conversation source: {path}")
    return rows


def _catalog(rows: list[Conversation]) -> list[ToolSpec]:
    by_name: dict[str, ToolSpec] = {}
    for row in rows:
        for tool in row.tools:
            by_name.setdefault(tool.name, tool)
    return [by_name[name] for name in sorted(by_name)]


def _features(model: LocalAgentLM, tokenizer: Any, decisions: list[ProbeDecision]) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [
                _feat(model, tokenizer, decision.prompt, "cpu", framed=decision.framed)
                for decision in decisions
            ]
        )


def _new_selector(
    model: LocalAgentLM,
    parent: dict[str, Any],
    *,
    warm: bool,
) -> DenseToolSelector:
    state = parent["dense_selector"]
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=int(state["t_proj.weight"].shape[1]),
        proj=int(state["q_proj.weight"].shape[0]),
    )
    if warm:
        selector.load_state_dict(state)
    return selector


def _train(
    model: LocalAgentLM,
    tokenizer: Any,
    parent: dict[str, Any],
    tools: list[ToolSpec],
    decisions: list[ProbeDecision],
    *,
    warm: bool,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[DenseToolSelector, dict[str, Any]]:
    selector = _new_selector(model, parent, warm=warm)
    state = parent["dense_selector"]
    embeddings = tool_embeddings(tools, int(state["t_proj.weight"].shape[1]))
    name_to_index = {tool.name: index for index, tool in enumerate(tools)}
    labels = torch.tensor([name_to_index[decision.ref_name] for decision in decisions])
    features = _features(model, tokenizer, decisions)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr)
    rng = random.Random(SEED)
    for _ in range(steps):
        indices = torch.tensor([rng.randrange(len(decisions)) for _ in range(batch_size)])
        loss = F.cross_entropy(selector(features[indices], embeddings), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = selector(features, embeddings)
        top1 = float((scores.argmax(-1) == labels).float().mean())
        top3 = float((scores.topk(min(3, len(tools)), dim=-1).indices == labels[:, None]).any(-1).float().mean())
    return selector, {"rows": len(decisions), "top1": top1, "top3": top3}


def _score(
    model: LocalAgentLM,
    tokenizer: Any,
    selector: DenseToolSelector,
    tools: list[ToolSpec],
    decisions: list[ProbeDecision],
    emb_dim: int,
) -> dict[str, Any]:
    features = _features(model, tokenizer, decisions)
    embeddings = tool_embeddings(tools, emb_dim)
    names = [tool.name for tool in tools]
    targets = torch.tensor([names.index(decision.ref_name) for decision in decisions])
    with torch.no_grad():
        scores = selector(features, embeddings)
        ranking = scores.argsort(dim=-1, descending=True)
    return {
        "rows": len(decisions),
        "top1": float((ranking[:, 0] == targets).float().mean()),
        "top3": float((ranking[:, : min(3, len(tools))] == targets[:, None]).any(-1).float().mean()),
    }


def _relative_movement(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, names in {
        "query_tower": ("q_proj.weight", "q_proj.bias"),
        "tool_tower": ("t_proj.weight", "t_proj.bias"),
    }.items():
        base = torch.cat([before[name].float().flatten() for name in names])
        delta = torch.cat([(after[name].float() - before[name].float()).flatten() for name in names])
        result[group] = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--backbone-init", choices=("warm", "random"), default="warm")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite selector-repair outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and lr must be positive")

    train_rows = _load(args.train)
    eval_rows = _load(args.eval)
    train_decisions = probe_decisions(train_rows)
    eval_decisions = probe_decisions(eval_rows)
    tools = _catalog(train_rows + eval_rows)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    if args.backbone_init == "random":
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
    else:
        model.load_state_dict(parent["state_dict"])
    model.eval()
    tokenizer_meta = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(tokenizer_meta["kind"], tokenizer_meta.get("path"))
    state = parent["dense_selector"]
    emb_dim = int(state["t_proj.weight"].shape[1])
    selector = _new_selector(model, parent, warm=args.backbone_init == "warm")
    before = _score(model, tokenizer, selector, tools, eval_decisions, emb_dim)
    selector, train_metrics = _train(
        model,
        tokenizer,
        parent,
        tools,
        train_decisions,
        warm=args.backbone_init == "warm",
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    after = _score(model, tokenizer, selector, tools, eval_decisions, emb_dim)
    child = dict(parent)
    child.update(
        {
            "dense_selector": selector.state_dict(),
            "selector_repair": {
                "dataset": "apple/ToolSandbox",
                "source_revision": SOURCE_REVISION,
                "backbone_frozen": True,
                "tool_catalog": [tool.name for tool in tools],
                "backbone_init": args.backbone_init,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_toolsandbox_dispatch_selector_repair",
        "schema_version": 1,
        "source": {
            "dataset": "apple/ToolSandbox",
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "manifest": _identity(args.manifest),
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_decisions": len(train_decisions),
            "eval_decisions": len(eval_decisions),
            "tool_count": len(tools),
        },
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": SEED,
            "backbone_init": args.backbone_init,
            "backbone_frozen": True,
        },
        "before_eval": before,
        "train": train_metrics,
        "after_eval": after,
        "selector_movement": _relative_movement(parent["dense_selector"], selector.state_dict()),
        "decision": {
            "native_replay_required": True,
            "adoption": "pending_native_bridge",
            "claim_boundary": "Frozen-backbone ToolSandbox candidate-selector repair only; not an official benchmark score, native success, screenshot grounding, or external side effect.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
