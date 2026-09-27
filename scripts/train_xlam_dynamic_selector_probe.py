#!/usr/bin/env python3
"""Train and audit a tool selector over the public xLAM-derived tool catalog.

The xLAM derivative carries a different tool catalog on every row (and many tools are unseen in
the LocalAgent browser pool).  This probe therefore uses the two-tower selector with a deterministic
union of row-local ``ToolSpec`` objects.  It is a dispatch diagnostic only: it does not claim the
gated Salesforce split, generated argument exactness, native API effects, or a benchmark score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings, train_dense_selector
from localagent.data.schema import Conversation, ToolSpec
from localagent.train.stage_data import probe_decisions
from scripts.train_cross_surface_continuation import (
    _assert_disjoint,
    _checkpoint_tokenizer,
    _identity,
    _load_rows,
)
from localagent.agent.tool_head import _feat


def _tool_key(tool: ToolSpec) -> str:
    return json.dumps(
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
        sort_keys=True,
        separators=(",", ":"),
    )


def _union_tools(conversations: Iterable[Conversation]) -> tuple[list[ToolSpec], int]:
    """Return name-sorted tools and count names with conflicting row-local schemas."""

    by_name: dict[str, set[str]] = {}
    by_key: dict[str, ToolSpec] = {}
    for conversation in conversations:
        for tool in conversation.tools:
            key = _tool_key(tool)
            by_name.setdefault(tool.name, set()).add(key)
            by_key.setdefault(key, tool)
    selected: list[ToolSpec] = []
    conflicts = 0
    for name in sorted(by_name):
        keys = sorted(by_name[name])
        if len(keys) > 1:
            conflicts += 1
        selected.append(by_key[keys[0]])
    return selected, conflicts


def _features(model: Any, tokenizer: Any, decisions: list[Any]) -> torch.Tensor:
    return torch.stack(
        [_feat(model, tokenizer, decision.prompt, "cpu", framed=decision.framed) for decision in decisions]
    )


def _metrics(
    model: Any,
    tokenizer: Any,
    selector: DenseToolSelector,
    candidate_tools: list[ToolSpec],
    conversations: list[Conversation],
    *,
    features: torch.Tensor | None = None,
    candidate_embs: torch.Tensor | None = None,
) -> dict[str, Any]:
    decisions = probe_decisions(conversations)
    if len(decisions) != len(conversations):
        raise ValueError("xLAM selector probe expects one assistant decision per conversation")
    features = _features(model, tokenizer, decisions) if features is None else features
    candidate_names = [tool.name for tool in candidate_tools]
    name_index = {name: index for index, name in enumerate(candidate_names)}
    embs = tool_embeddings(candidate_tools, selector.emb_dim) if candidate_embs is None else candidate_embs
    with torch.no_grad():
        scores = selector(features, embs)
    global_correct = 0
    row_local_correct = 0
    row_local_sizes: list[int] = []
    unseen = 0
    for row_index, (conversation, decision) in enumerate(zip(conversations, decisions, strict=True)):
        gold = decision.ref_name
        if gold not in name_index:
            raise ValueError(f"gold tool {gold!r} missing from candidate union")
        global_correct += int(candidate_names[int(scores[row_index].argmax())] == gold)
        local_names = [tool.name for tool in conversation.tools]
        local_indices = [name_index[name] for name in local_names if name in name_index]
        if gold not in local_names:
            raise ValueError(f"gold tool {gold!r} missing from row-local catalog")
        local_scores = scores[row_index, local_indices]
        row_local_correct += int(local_names[int(local_scores.argmax())] == gold)
        row_local_sizes.append(len(local_indices))
        unseen += int(gold not in {tool.name for tool in candidate_tools})
    rows = len(decisions)
    return {
        "rows": rows,
        "global_tool_top1": global_correct / rows if rows else 0.0,
        "row_local_tool_top1": row_local_correct / rows if rows else 0.0,
        "row_local_candidate_count_mean": sum(row_local_sizes) / rows if rows else 0.0,
        "row_local_candidate_count_max": max(row_local_sizes, default=0),
        "gold_tools_unseen_in_candidate_union": unseen,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--eval-data", type=Path, action="append", required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5.0e-3)
    parser.add_argument("--proj", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite selector-probe outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.proj < 1:
        raise SystemExit("steps, batch-size, and proj must be positive; lr must be positive")

    train_rows = _load_rows(args.data)
    eval_rows = _load_rows(args.eval_data)
    _assert_disjoint(train_rows, eval_rows)
    all_rows = train_rows + eval_rows
    train_tools, train_conflicts = _union_tools(train_rows)
    candidate_tools, candidate_conflicts = _union_tools(all_rows)
    train_decisions = probe_decisions(train_rows)
    eval_decisions = probe_decisions(eval_rows)
    if not train_decisions:
        raise SystemExit("no tool decisions in training rows")

    parent_identity = _identity(args.init)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    from localagent.model import LocalAgentLM, ModelConfig

    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)
    # Backbone features and the fixed char-ngram candidate matrix are reused across all four
    # before/after metrics; recomputing them would turn this diagnostic into several thousand
    # redundant decoder passes.
    train_features = _features(model, tokenizer, train_decisions)
    eval_features = _features(model, tokenizer, eval_decisions)
    candidate_embs = tool_embeddings(candidate_tools, 8192)
    torch.manual_seed(args.seed)
    selector = DenseToolSelector(model.cfg.d_model, emb_dim=8192, proj=args.proj)
    before_train = _metrics(
        model, tokenizer, selector, candidate_tools, train_rows,
        features=train_features, candidate_embs=candidate_embs,
    )
    before_eval = _metrics(
        model, tokenizer, selector, candidate_tools, eval_rows,
        features=eval_features, candidate_embs=candidate_embs,
    )
    selector = train_dense_selector(
        model,
        train_decisions,
        tokenizer,
        train_tools,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        proj=args.proj,
        device="cpu",
        log=print,
    )
    after_train = _metrics(
        model, tokenizer, selector, candidate_tools, train_rows,
        features=train_features, candidate_embs=candidate_embs,
    )
    after_eval = _metrics(
        model, tokenizer, selector, candidate_tools, eval_rows,
        features=eval_features, candidate_embs=candidate_embs,
    )

    child = dict(parent)
    child.update(
        {
            "dense_selector": selector.state_dict(),
            "stage": "sft_xlam_dynamic_selector_probe",
            "parent_checkpoint_sha256": parent_identity["sha256"],
            "xlam_dynamic_selector": {
                "train_tools": len(train_tools),
                "candidate_tools": len(candidate_tools),
                "train_schema_conflicts": train_conflicts,
                "candidate_schema_conflicts": candidate_conflicts,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "proj": args.proj,
                "seed": args.seed,
                "backbone_frozen": True,
                "before_train": before_train,
                "before_eval": before_eval,
                "after_train": after_train,
                "after_eval": after_eval,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    child_identity = _identity(args.output)
    report = {
        "kind": "localagent_xlam_dynamic_selector_report",
        "schema_version": 1,
        "source": {
            "dataset": "product-science/xlam-function-calling-60k-raw",
            "url": "https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw",
            "revision": "dfbd3c669354c27f2727870d39a4d86c32381448",
            "license": "apache-2.0",
            "official_salesforce_split": False,
        },
        "parent": parent_identity,
        "child": child_identity,
        "rows": {"train": len(train_rows), "eval": len(eval_rows), "train_decisions": len(train_decisions)},
        "tools": {
            "train_union": len(train_tools),
            "candidate_union": len(candidate_tools),
            "train_schema_conflicts": train_conflicts,
            "candidate_schema_conflicts": candidate_conflicts,
        },
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "proj": args.proj,
            "seed": args.seed,
            "backbone_frozen": True,
        },
        "before_train": before_train,
        "before_eval": before_eval,
        "after_train": after_train,
        "after_eval": after_eval,
        "claim_boundary": (
            "Frozen-backbone two-tower dispatch probe over a public xLAM-derived candidate catalog; "
            "not the gated Salesforce split, generated argument exactness, native API execution, "
            "or an official benchmark score. Global metrics include eval-row tool descriptions in "
            "the candidate catalog; row-local metrics are the more favorable closed-world diagnostic."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
