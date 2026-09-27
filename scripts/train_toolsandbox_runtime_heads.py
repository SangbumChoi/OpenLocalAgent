#!/usr/bin/env python3
"""Train route and candidate-aware selector probes on ToolSandbox runtime ToolSpecs.

The input rows are the public source projection after replacing static schemas with the exact
``ExecutionContext.get_available_tools`` catalog for each named scenario.  The language-model
backbone stays frozen; only the route head and dense two-tower selector move.  This is a bounded
transfer diagnostic, not an official ToolSandbox score: the simulator state, user simulator, and
official verifier are deliberately outside this script.
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

SOURCE_URL = "https://github.com/apple/ToolSandbox"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
SEED = 2047


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
    with torch.no_grad():
        return torch.stack(
            [_feat(model, tokenizer, d.prompt, "cpu", framed=d.framed) for d in decisions]
        )


def _new_selector(model: LocalAgentLM, parent: dict[str, Any], *, warm: bool) -> DenseToolSelector:
    state = parent["dense_selector"]
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=int(state["t_proj.weight"].shape[1]),
        proj=int(state["q_proj.weight"].shape[0]),
    )
    if warm:
        selector.load_state_dict(state)
    return selector


def _candidate_embeddings(rows: list[Conversation], emb_dim: int) -> list[torch.Tensor]:
    return [tool_embeddings(row.tools, emb_dim) for row in rows]


def _metrics(
    selector: DenseToolSelector,
    route: RouteHead,
    features: torch.Tensor,
    rows: list[Conversation],
    decisions: list[ProbeDecision],
    embeddings: list[torch.Tensor],
) -> dict[str, float | int]:
    if len(rows) != len(decisions) or len(rows) != len(embeddings):
        raise ValueError("runtime projection must contain one simple decision per row")
    selector_hits = []
    selector_top3 = []
    route_hits = []
    with torch.no_grad():
        route_scores = route(features)
        for index, (row, decision, embs) in enumerate(zip(rows, decisions, embeddings)):
            names = [tool.name for tool in row.tools]
            if decision.kind != "tool" or decision.ref_name not in names:
                raise ValueError(f"gold tool is absent from runtime candidate list: {decision.ref_name}")
            scores = selector(features[index : index + 1], embs)[0]
            target = names.index(decision.ref_name)
            order = scores.argsort(descending=True)
            selector_hits.append(int(order[0]) == target)
            selector_top3.append(target in order[: min(3, len(names))].tolist())
            route_hits.append(int(route_scores[index].argmax()) == ROUTE_INDEX["app_action"])
    return {
        "rows": len(rows),
        "selector_top1": float(np.mean(selector_hits)),
        "selector_top3": float(np.mean(selector_top3)),
        "route_app_action": float(np.mean(route_hits)),
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


def _head_movement(before: dict[str, Any], after: dict[str, Any]) -> float:
    base = torch.cat([value.float().flatten() for value in before.values()])
    delta = torch.cat([(after[key].float() - before[key].float()).flatten() for key in before])
    return float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--backbone-init", choices=("warm", "random"), default="warm")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite runtime-head outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and lr must be positive")

    train_rows = _load(args.train)
    eval_rows = _load(args.eval)
    train_decisions = probe_decisions(train_rows)
    eval_decisions = probe_decisions(eval_rows)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    if args.backbone_init == "random":
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
    model = LocalAgentLM(config)
    if args.backbone_init == "warm":
        model.load_state_dict(parent["state_dict"])
    model.eval()
    tokenizer_meta = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(tokenizer_meta["kind"], tokenizer_meta.get("path"))
    train_features = _features(model, tokenizer, train_decisions)
    eval_features = _features(model, tokenizer, eval_decisions)
    emb_dim = int(parent["dense_selector"]["t_proj.weight"].shape[1])
    train_embeddings = _candidate_embeddings(train_rows, emb_dim)
    eval_embeddings = _candidate_embeddings(eval_rows, emb_dim)
    selector = _new_selector(model, parent, warm=args.backbone_init == "warm")
    route = RouteHead(model.cfg.d_model)
    if args.backbone_init == "warm":
        route.load_state_dict(parent["route_head"])
    before = _metrics(selector, route, eval_features, eval_rows, eval_decisions, eval_embeddings)
    optimizer = torch.optim.AdamW(list(selector.parameters()) + list(route.parameters()), lr=args.lr)
    rng = random.Random(SEED)
    labels = torch.full((len(train_rows),), ROUTE_INDEX["app_action"], dtype=torch.long)
    for _ in range(args.steps):
        indices = [rng.randrange(len(train_rows)) for _ in range(args.batch_size)]
        route_loss = F.cross_entropy(route(train_features[indices]), labels[indices])
        selector_losses = []
        for index in indices:
            names = [tool.name for tool in train_rows[index].tools]
            target = names.index(train_decisions[index].ref_name)
            selector_losses.append(
                F.cross_entropy(selector(train_features[index : index + 1], train_embeddings[index]),
                                torch.tensor([target]))
            )
        loss = route_loss + torch.stack(selector_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = _metrics(selector, route, eval_features, eval_rows, eval_decisions, eval_embeddings)
    child = dict(parent)
    child.update(
        {
            "dense_selector": selector.state_dict(),
            "route_head": route.state_dict(),
            "runtime_toolsandbox_head_training": {
                "source": "apple/ToolSandbox",
                "source_revision": SOURCE_REVISION,
                "runtime_candidate_catalog": True,
                "backbone_frozen": True,
                "backbone_init": args.backbone_init,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_toolsandbox_runtime_heads",
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
            "joint_heads": ["route_head", "dense_selector"],
        },
        "before_eval": before,
        "after_eval": after,
        "selector_movement": _relative_movement(parent["dense_selector"], selector.state_dict()),
        "route_movement": _head_movement(parent["route_head"], route.state_dict()),
        "decision": {
            "native_replay_required": True,
            "adoption": "pending_native_bridge",
            "claim_boundary": "Frozen-backbone route and row-local candidate-selector transfer only; no official split, user simulator, external API, or screenshot-grounding score.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
