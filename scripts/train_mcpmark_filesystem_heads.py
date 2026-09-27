#!/usr/bin/env python3
"""Adapt only route/selector heads on public MCPMark filesystem trajectories.

This is a frozen-backbone diagnostic.  It uses the real public MCPMark candidate tool catalogs,
keeps the language model tensors unchanged, and is still evaluated separately from the native
MCP server/verifier.  The output checkpoint is suitable for a subsequent isolated replay.
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
from localagent.agent.routes import ROUTE_INDEX, RouteHead
from localagent.agent.tool_head import _feat
from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import ProbeDecision, probe_decisions

SOURCE_URL = "https://github.com/eval-sys/mcpmark"
SOURCE_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"
SEED = 2049


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


def _features(model: LocalAgentLM, tokenizer: Any, decisions: list[ProbeDecision]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.stack([_feat(model, tokenizer, item.prompt, "cpu", framed=item.framed) for item in decisions])


def _metrics(
    selector: DenseToolSelector,
    route: RouteHead,
    features: torch.Tensor,
    rows: list[Conversation],
    decisions: list[ProbeDecision],
    embeddings: list[torch.Tensor],
) -> dict[str, float | int]:
    selector_hits: list[bool] = []
    selector_top3: list[bool] = []
    route_hits: list[bool] = []
    with torch.no_grad():
        route_scores = route(features)
        for index, (row, decision, candidate_embeddings) in enumerate(zip(rows, decisions, embeddings)):
            names = [tool.name for tool in row.tools]
            if decision.kind != "tool" or decision.ref_name not in names:
                raise ValueError(f"gold tool is absent from candidate list: {decision.ref_name}")
            target = names.index(decision.ref_name)
            order = selector(features[index : index + 1], candidate_embeddings)[0].argsort(descending=True)
            selector_hits.append(int(order[0]) == target)
            selector_top3.append(target in order[: min(3, len(names))].tolist())
            route_hits.append(int(route_scores[index].argmax()) == ROUTE_INDEX["app_action"])
    return {
        "rows": len(rows),
        "selector_top1": float(np.mean(selector_hits)),
        "selector_top3": float(np.mean(selector_top3)),
        "route_app_action": float(np.mean(route_hits)),
    }


def _relative_l2(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    base = torch.cat([value.float().flatten() for value in before.values()])
    delta = torch.cat([(after[name].float() - before[name].float()).flatten() for name in before])
    return float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base).clamp_min(1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite head-adaptation outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and lr must be positive")
    train_rows = _load(args.train)
    eval_rows = _load(args.eval)
    train_decisions = probe_decisions(train_rows)
    eval_decisions = probe_decisions(eval_rows)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    model.eval()
    metadata = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(metadata["kind"], metadata.get("path"))
    train_features = _features(model, tokenizer, train_decisions)
    eval_features = _features(model, tokenizer, eval_decisions)
    emb_dim = int(parent["dense_selector"]["t_proj.weight"].shape[1])
    train_embeddings = [tool_embeddings(row.tools, emb_dim) for row in train_rows]
    eval_embeddings = [tool_embeddings(row.tools, emb_dim) for row in eval_rows]
    selector = DenseToolSelector(model.cfg.d_model, emb_dim=emb_dim, proj=int(parent["dense_selector"]["q_proj.weight"].shape[0]))
    selector.load_state_dict(parent["dense_selector"])
    route = RouteHead(model.cfg.d_model)
    route.load_state_dict(parent["route_head"])
    before_selector = {name: value.detach().clone() for name, value in selector.state_dict().items()}
    before_route = {name: value.detach().clone() for name, value in route.state_dict().items()}
    before = _metrics(selector, route, eval_features, eval_rows, eval_decisions, eval_embeddings)
    optimizer = torch.optim.AdamW(list(selector.parameters()) + list(route.parameters()), lr=args.lr)
    rng = random.Random(SEED)
    for step in range(args.steps):
        indices = [rng.randrange(len(train_rows)) for _ in range(min(args.batch_size, len(train_rows)))]
        route_target = torch.full((len(indices),), ROUTE_INDEX["app_action"], dtype=torch.long)
        route_loss = F.cross_entropy(route(train_features[indices]), route_target)
        selector_losses = []
        for index in indices:
            names = [tool.name for tool in train_rows[index].tools]
            target = names.index(train_decisions[index].ref_name)
            selector_losses.append(
                F.cross_entropy(
                    selector(train_features[index : index + 1], train_embeddings[index]),
                    torch.tensor([target]),
                )
            )
        loss = route_loss + torch.stack(selector_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"[mcpmark-heads] step {step}/{args.steps} loss {loss.item():.4f}", flush=True)
    after = _metrics(selector, route, eval_features, eval_rows, eval_decisions, eval_embeddings)
    child = dict(parent)
    child.update(
        {
            "dense_selector": selector.state_dict(),
            "route_head": route.state_dict(),
            "stage": "mcpmark_filesystem_frozen_head_adaptation",
            "mcpmark_filesystem_head_adaptation": {
                "source_revision": SOURCE_REVISION,
                "backbone_frozen": True,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_mcpmark_filesystem_head_adaptation",
        "schema_version": 1,
        "source": {
            "dataset": "MCPMark",
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr, "seed": SEED, "backbone_frozen": True},
        "before_eval": before,
        "after_eval": after,
        "head_movement": {"dense_selector_relative_l2": _relative_l2(before_selector, selector.state_dict()), "route_relative_l2": _relative_l2(before_route, route.state_dict())},
        "claim_boundary": "Frozen-backbone route/selector adaptation on public MCPMark filesystem trajectories; native replay and the official split remain separate, and no external-account or leaderboard score is claimed.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
