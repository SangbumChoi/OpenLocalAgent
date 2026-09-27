#!/usr/bin/env python3
"""Adapt frozen route/selector heads on a cross-surface public continuation child.

This probe keeps the language-model backbone fixed and trains the same route and dense-selector
budgets for a warm-start or random-backbone child.  Inputs are source-labelled normalized
``Conversation`` JSONL files; the report is a text-first tool-use diagnostic, not native task
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.routes import train_route_head
from localagent.agent.toolset import REALISTIC_BROWSER_TOOLS
from localagent.train.stage_data import probe_decisions
from scripts.analyze_weight_transfer import analyze as analyze_weight_transfer
from scripts.train_cross_surface_continuation import (
    _assert_source_disjoint,
    _checkpoint_tokenizer,
    _identity,
    _load_labeled_groups,
    _parse_labeled_path,
    _parse_source_reference,
    _source_profile,
)
from scripts.train_public_agent_continuation import _head_metrics, _load_heads


def _metrics_by_source(
    groups: list[tuple[str, Path, list[Any]]],
    model: Any,
    tokenizer: Any,
    route: Any,
    selector: BoundSelector,
) -> dict[str, Any]:
    return {
        label: _head_metrics(model, tokenizer, route, selector, rows)
        for label, _path, rows in groups
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument("--eval-data", type=_parse_labeled_path, action="append", required=True)
    parser.add_argument(
        "--source-reference",
        type=_parse_source_reference,
        action="append",
        default=[],
        help="public source identity as LABEL=DATASET|URL; repeat for every source",
    )
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--head-steps", type=int, default=200)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=5.0e-3)
    parser.add_argument("--head-seed", type=int, default=2029)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite head-probe outputs")
    if args.head_steps < 1 or args.head_batch_size < 1 or args.head_lr <= 0:
        raise SystemExit("head-steps and head-batch-size must be positive; head-lr must be positive")

    train_groups = _load_labeled_groups(args.data)
    eval_groups = _load_labeled_groups(args.eval_data)
    references = dict(args.source_reference)
    expected_labels = {label for label, _path, _rows in train_groups} | {
        label for label, _path, _rows in eval_groups
    }
    if set(references) != expected_labels:
        missing = sorted(expected_labels - set(references))
        extra = sorted(set(references) - expected_labels)
        raise SystemExit(f"source references must match labels; missing={missing}, extra={extra}")
    train_rows = [row for _label, _path, rows in train_groups for row in rows]
    eval_rows = [row for _label, _path, rows in eval_groups for row in rows]
    _assert_source_disjoint(train_groups, eval_groups)

    parent_identity = _identity(args.init)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    from localagent.model import LocalAgentLM, ModelConfig

    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)
    route, selector = _load_heads(parent, model, init="parent", seed=args.head_seed)
    before = _metrics_by_source(eval_groups, model, tokenizer, route, selector)
    decisions = probe_decisions(train_rows)
    torch.manual_seed(args.head_seed)
    route = train_route_head(
        model,
        decisions,
        tokenizer,
        steps=args.head_steps,
        batch_size=args.head_batch_size,
        lr=args.head_lr,
        device=args.device,
        log=print,
    )
    selector_model = train_dense_selector(
        model,
        decisions,
        tokenizer,
        REALISTIC_BROWSER_TOOLS,
        steps=args.head_steps,
        batch_size=args.head_batch_size,
        lr=args.head_lr,
        device=args.device,
        proj=int(parent["selector_proj"]),
        examples=parent.get("examples", {}),
        log=print,
    )
    selector = BoundSelector(selector_model, REALISTIC_BROWSER_TOOLS, examples=parent.get("examples", {}))
    after = _metrics_by_source(eval_groups, model, tokenizer, route, selector)

    child = dict(parent)
    child.update(
        {
            "route_head": route.state_dict(),
            "dense_selector": selector.model.state_dict(),
            "stage": "sft_cross_surface_dispatch_head_probe",
            "parent_checkpoint_sha256": parent_identity["sha256"],
            "cross_surface_head_training": {
                "train_sources": [
                    _source_profile(label, path, rows, references[label])
                    for label, path, rows in train_groups
                ],
                "eval_sources": [
                    _source_profile(label, path, rows, references[label])
                    for label, path, rows in eval_groups
                ],
                "head_steps": args.head_steps,
                "head_batch_size": args.head_batch_size,
                "head_lr": args.head_lr,
                "head_seed": args.head_seed,
                "backbone_frozen": True,
                "before_eval_by_source": before,
                "after_eval_by_source": after,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    transfer = analyze_weight_transfer(args.init, args.output)
    report = {
        "kind": "localagent_cross_surface_dispatch_head_report",
        "schema_version": 1,
        "parent": parent_identity,
        "child": _identity(args.output),
        "train_sources": [
            _source_profile(label, path, rows, references[label])
            for label, path, rows in train_groups
        ],
        "eval_sources": [
            _source_profile(label, path, rows, references[label])
            for label, path, rows in eval_groups
        ],
        "rows": {"train": len(train_rows), "eval": len(eval_rows), "train_decisions": len(decisions)},
        "hyperparameters": {
            "head_steps": args.head_steps,
            "head_batch_size": args.head_batch_size,
            "head_lr": args.head_lr,
            "head_seed": args.head_seed,
            "device": args.device,
            "backbone_frozen": True,
        },
        "before_eval_by_source": before,
        "after_eval_by_source": after,
        "weight_transfer": {
            "compatibility": transfer["compatibility"],
            "groups": transfer["groups"],
        },
        "claim_boundary": (
            "Frozen-backbone route/selector adaptation on public-train-only text/accessibility rows; "
            "not an official benchmark score, native environment success, screenshot grounding, "
            "or evidence of real email/Notion/MCP side effects."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
