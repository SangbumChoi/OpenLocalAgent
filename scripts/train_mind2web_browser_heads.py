#!/usr/bin/env python3
"""Train browser route/selector heads from public Mind2Web DOM/action rows.

The language-model body stays frozen.  Mind2Web's ``web_click``, ``web_type``, and ``web_select``
labels are mapped to the deployed standard ``click``/``type_text`` vocabulary, while the original
DOM/task prompts remain the feature input.  This is a source-disjoint browser-head transfer
diagnostic; it is not BrowserGym training and never reads BrowserGym goals or verifiers.
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
from localagent.agent.tool_head import canonical_tool_name, _feat
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.agent_synth import Sample
from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import probe_decisions

SEED = 2047


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _rows(path: Path) -> list[Conversation]:
    rows = [Conversation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty Mind2Web source: {path}")
    return rows


def _samples(rows: list[Conversation]) -> list[Sample]:
    samples: list[Sample] = []
    for decision in probe_decisions(rows):
        if decision.kind != "tool":
            continue
        name = canonical_tool_name(decision.ref_name)
        if name not in {tool.name for tool in STANDARD_TOOLS}:
            raise ValueError(f"unsupported Mind2Web action: {decision.ref_name}")
        sample = Sample(
            "mind2web_browser",
            "computer_use",
            decision.prompt,
            "tool",
            json.dumps({"name": name, "arguments": {}}, separators=(",", ":")),
            name,
            "{}",
        )
        setattr(sample, "framed", decision.framed)
        samples.append(sample)
    if not samples:
        raise ValueError("Mind2Web source has no browser tool decisions")
    return samples


def _features(model: LocalAgentLM, tokenizer: Any, samples: list[Sample]) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [_feat(model, tokenizer, sample.prompt, "cpu", framed=sample.framed) for sample in samples]
        )


def _metrics(
    route: RouteHead,
    selector: DenseToolSelector,
    route_features: torch.Tensor,
    selector_features: torch.Tensor,
    samples: list[Sample],
    embeddings: torch.Tensor,
) -> dict[str, float | int]:
    if len(route_features) != len(samples) or len(selector_features) != len(samples):
        raise ValueError("feature/sample count mismatch")
    with torch.no_grad():
        route_pred = route(route_features).argmax(-1)
        scores = selector(selector_features, embeddings)
    route_ok = route_pred.eq(ROUTE_INDEX["computer_use"]).tolist()
    names = [tool.name for tool in STANDARD_TOOLS]
    targets = [names.index(sample.ref_name) for sample in samples]
    order = scores.argsort(dim=-1, descending=True)
    top1 = [int(row[0]) == target for row, target in zip(order, targets)]
    top3 = [target in row[:3].tolist() for row, target in zip(order, targets)]
    return {
        "rows": len(samples),
        "route_accuracy": float(np.mean(route_ok)),
        "selector_top1": float(np.mean(top1)),
        "selector_top3": float(np.mean(top3)),
    }


def _movement(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    base = torch.cat([value.float().flatten() for value in before.values()])
    delta = torch.cat([(after[key].float() - before[key].float()).flatten() for key in before])
    return float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--backbone-init", choices=("warm", "random"), default="warm")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite browser-head outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and lr must be positive")

    train_samples = _samples(_rows(args.train))
    eval_samples = _samples(_rows(args.eval))
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    model.eval()
    tokenizer_meta = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(tokenizer_meta["kind"], tokenizer_meta.get("path"))
    if args.backbone_init == "random":
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    train_features = _features(model, tokenizer, train_samples)
    eval_features = _features(model, tokenizer, eval_samples)
    embeddings = tool_embeddings(STANDARD_TOOLS, dim=int(parent["dense_selector"]["t_proj.weight"].shape[1]))
    route = RouteHead(config.d_model)
    selector = DenseToolSelector(
        config.d_model,
        emb_dim=int(parent["dense_selector"]["t_proj.weight"].shape[1]),
        proj=int(parent["dense_selector"]["q_proj.weight"].shape[0]),
    )
    if args.backbone_init == "warm":
        route.load_state_dict(parent["route_head"])
        selector.load_state_dict(parent["dense_selector"])
    before = _metrics(route, selector, eval_features, eval_features, eval_samples, embeddings)
    route_labels = torch.full((len(train_samples),), ROUTE_INDEX["computer_use"], dtype=torch.long)
    names = [tool.name for tool in STANDARD_TOOLS]
    selector_labels = torch.tensor([names.index(sample.ref_name) for sample in train_samples], dtype=torch.long)
    optimizer = torch.optim.AdamW(list(route.parameters()) + list(selector.parameters()), lr=args.lr)
    rng = random.Random(SEED)
    for _ in range(args.steps):
        indices = [rng.randrange(len(train_samples)) for _ in range(args.batch_size)]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        route_loss = F.cross_entropy(route(train_features[index_tensor]), route_labels[index_tensor])
        selector_loss = F.cross_entropy(
            selector(train_features[index_tensor], embeddings), selector_labels[index_tensor]
        )
        loss = route_loss + selector_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = _metrics(route, selector, eval_features, eval_features, eval_samples, embeddings)
    child = dict(parent)
    child.update(
        {
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "browser_mind2web_head_training": {
                "source": "osunlp/Mind2Web",
                "backbone_frozen": True,
                "backbone_init": args.backbone_init,
                "canonical_actions": {"web_click": "click", "web_select": "click", "web_type": "type_text"},
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_mind2web_browser_head_transfer",
        "schema_version": 1,
        "source": {
            "dataset": "osunlp/Mind2Web",
            "url": "https://huggingface.co/datasets/osunlp/Mind2Web",
            "revision": "17ece8eb89862368edc0cc806acee6fca5163474",
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "train_rows": len(train_samples),
            "eval_rows": len(eval_samples),
        },
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr, "seed": SEED, "backbone_frozen": True, "backbone_init": args.backbone_init},
        "before_eval": before,
        "after_eval": after,
        "route_movement": _movement(parent["route_head"], route.state_dict()),
        "selector_movement": _movement(parent["dense_selector"], selector.state_dict()),
        "decision": {"native_replay_required": True, "adoption": "pending_native_browsergym_canary", "claim_boundary": "Frozen-backbone route and dense-selector transfer over public Mind2Web train rows only; no BrowserGym task prompts, verifiers, screenshot grounding, or external side effects."},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
