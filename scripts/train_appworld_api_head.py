#!/usr/bin/env python3
"""Train a frozen-backbone AppWorld app.api schema head on public train rows.

The resulting sidecar is intentionally small and only restricts the existing native evaluator's
schema candidates. It does not claim a complete AppWorld policy or a leaderboard score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from localagent.data.schema import Conversation
from localagent.eval.appworld_api_head import (
    _features,
    first_action_examples,
    head_metrics,
    save_appworld_api_head,
    train_appworld_api_head,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _rows(path: Path) -> list[Conversation]:
    rows = [
        Conversation.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no Conversation rows in {path}")
    return rows


def _assert_disjoint(train: list[Conversation], evaluation: list[Conversation]) -> None:
    train_ids = {
        str(row.meta.get("parent_record_id"))
        for row in train
        if row.meta.get("parent_record_id")
    }
    eval_ids = {
        str(row.meta.get("parent_record_id"))
        for row in evaluation
        if row.meta.get("parent_record_id")
    }
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"train/eval parent_record_id overlap: {sorted(overlap)[:5]}")


def _tokenizer(parent: dict[str, Any]):
    metadata = parent.get("tokenizer") or {"kind": "byte"}
    kind = str(metadata.get("kind", "byte"))
    path = metadata.get("path")
    return load_tokenizer(kind, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-dataset", default="appworld")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite API-head output or report")

    train_rows = _rows(args.data)
    eval_rows = _rows(args.eval_data)
    _assert_disjoint(train_rows, eval_rows)
    parent_identity = _identity(args.init)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    tokenizer = _tokenizer(parent)
    train_prompts, train_labels = first_action_examples(train_rows)
    eval_prompts, eval_labels = first_action_examples(eval_rows)
    if not set(eval_labels) <= set(train_labels):
        raise ValueError("evaluation contains an API label absent from public train rows")
    head, train_metrics = train_appworld_api_head(
        model,
        tokenizer,
        train_prompts,
        train_labels,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        log=print,
    )
    eval_metrics = head_metrics(
        head,
        _features(model, tokenizer, eval_prompts, args.device),
        eval_labels,
    )
    source = {
        "dataset": args.source_dataset,
        "revision": args.source_revision,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_input": _identity(args.data),
        "eval_input": _identity(args.eval_data),
    }
    save_appworld_api_head(
        args.output,
        head,
        parent_checkpoint=parent_identity,
        source=source,
    )
    report = {
        "kind": "localagent_appworld_api_head_training_report",
        "schema_version": 1,
        "source": source,
        "parent": parent_identity,
        "child": _identity(args.output),
        "classes": list(head.classes),
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": args.device,
            "seed": 2027,
        },
        "metrics": {"train": train_metrics, "eval": eval_metrics},
        "claim_boundary": (
            "Frozen-backbone AppWorld app.api schema head trained on public train rows and measured "
            "on disjoint dev rows. This sidecar restricts bounded schema replay; it is not a complete "
            "AppWorld trajectory, official leaderboard score, or external-account result."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
