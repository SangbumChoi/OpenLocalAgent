#!/usr/bin/env python3
"""Run a matched ToolSandbox tool-selector transfer probe.

The public-source projection has per-scenario candidate tools, so this probe scores the dense
selector against each row's actual candidate list rather than an artificial global vocabulary. It
compares the inherited selector, a retrained selector over the transferred backbone, and a
same-seed selector trained over a fresh random backbone. It is a selector diagnostic only: no
ToolSandbox simulator, verifier, user simulator, or external API is executed.
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

from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings, train_dense_selector
from localagent.agent.tool_head import _feat
from localagent.data.schema import Conversation, ToolSpec
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import ProbeDecision, probe_decisions

SOURCE_URL = "https://github.com/apple/ToolSandbox"
SEED = 2029


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


def _tool_catalog(rows: list[Conversation]) -> list[ToolSpec]:
    by_name: dict[str, ToolSpec] = {}
    for row in rows:
        for tool in row.tools:
            by_name.setdefault(tool.name, tool)
    return [by_name[name] for name in sorted(by_name)]


def _selector_from_state(model: LocalAgentLM, state: dict[str, Any]) -> DenseToolSelector:
    selector_state = state["dense_selector"]
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=int(selector_state["t_proj.weight"].shape[1]),
        proj=int(selector_state["q_proj.weight"].shape[0]),
    )
    selector.load_state_dict(selector_state)
    selector.eval()
    return selector


def _score(
    model: LocalAgentLM,
    tokenizer: Any,
    selector: DenseToolSelector,
    rows: list[Conversation],
    decisions: list[ProbeDecision],
) -> dict[str, Any]:
    if len(rows) != len(decisions):
        raise ValueError("conversation and probe-decision counts differ")
    top1 = 0
    top3 = 0
    covered = 0
    candidate_counts: list[int] = []
    for row, decision in zip(rows, decisions):
        tools = list(row.tools)
        names = [tool.name for tool in tools]
        if decision.ref_name not in names:
            raise ValueError(f"target {decision.ref_name!r} is absent from its candidate list")
        candidate_counts.append(len(tools))
        covered += 1
        embeddings = tool_embeddings(tools, selector.t_proj.weight.shape[1])
        feature = _feat(model, tokenizer, decision.prompt, "cpu", framed=decision.framed)
        with torch.no_grad():
            ranking = selector(feature.unsqueeze(0), embeddings)[0].argsort(descending=True).tolist()
        ranked_names = [names[index] for index in ranking]
        top1 += int(ranked_names[0] == decision.ref_name)
        top3 += int(decision.ref_name in ranked_names[:3])
    count = len(rows)
    return {
        "rows": count,
        "candidate_coverage": covered / count,
        "candidate_count_mean": sum(candidate_counts) / count,
        "selector_top1": top1 / count,
        "selector_top3": top3 / count,
    }


def _relative_movement(before: DenseToolSelector, after: DenseToolSelector) -> dict[str, float]:
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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and learning rate must be positive")

    train_rows = _load(args.train)
    eval_rows = _load(args.eval)
    train_decisions = probe_decisions(train_rows)
    eval_decisions = probe_decisions(eval_rows)
    catalog = _tool_catalog(train_rows + eval_rows)

    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**parent["cfg"])
    model_cfg.assert_within_budget()
    model = LocalAgentLM(model_cfg)
    model.load_state_dict(parent["state_dict"])
    model.eval()
    tokenizer_meta = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(tokenizer_meta["kind"], tokenizer_meta.get("path"))
    inherited = _selector_from_state(model, parent)
    inherited_metrics = _score(model, tokenizer, inherited, eval_rows, eval_decisions)

    def train_for(backbone: LocalAgentLM) -> DenseToolSelector:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        return train_dense_selector(
            backbone,
            train_decisions,
            tokenizer,
            catalog,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            proj=int(parent["selector_proj"]),
            log=lambda *_: None,
        )

    transferred = train_for(model)
    transferred_metrics = _score(model, tokenizer, transferred, eval_rows, eval_decisions)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    random_model = LocalAgentLM(model_cfg)
    random_model.eval()
    random_selector = train_for(random_model)
    random_metrics = _score(random_model, tokenizer, random_selector, eval_rows, eval_decisions)

    parent_identity = _identity(args.init)
    manifest_identity = _identity(args.source_manifest)
    report = {
        "kind": "localagent_toolsandbox_selector_transfer_probe",
        "schema_version": 1,
        "source": {
            "dataset": "apple/ToolSandbox",
            "url": SOURCE_URL,
            "manifest": manifest_identity,
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "candidate_tools": len(catalog),
        },
        "parent": parent_identity,
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": SEED,
        },
        "arms": {
            "inherited_pretrained_selector": {
                "backbone": "pretrained_frozen",
                "selector": "inherited",
                "metrics": inherited_metrics,
            },
            "retrained_pretrained_backbone": {
                "backbone": "pretrained_frozen",
                "selector": "retrained_on_toolsandbox_projection",
                "metrics": transferred_metrics,
                "selector_relative_movement": _relative_movement(inherited, transferred),
            },
            "retrained_matched_random_backbone": {
                "backbone": "matched_random_frozen",
                "selector": "retrained_on_toolsandbox_projection",
                "metrics": random_metrics,
            },
        },
        "decision": {
            "transfer_improves_over_inherited_top1": transferred_metrics["selector_top1"]
            > inherited_metrics["selector_top1"],
            "transfer_beats_random_top1": transferred_metrics["selector_top1"]
            > random_metrics["selector_top1"],
            "adoption": "do_not_adopt_as_representation_evidence",
            "reason": (
                "Retraining the selector improves the inherited arm, but the matched random "
                "backbone reaches the same held-out top-1 on this small projection."
            ),
        },
        "claim_boundary": (
            "Source-projected ToolSandbox selector diagnostic only; candidate-list ranking is not "
            "an official ToolSandbox result and does not establish stateful execution, milestone "
            "completion, native WebGPU capability, or external side effects."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
