#!/usr/bin/env python3
"""Build a public-train prompt retriever for bounded AppWorld schema grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from localagent.data.schema import Conversation
from localagent.eval.appworld_api_head import (
    AppWorldAPINearestNeighbor,
    first_action_argument_fields,
    first_action_examples,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=8192)
    parser.add_argument("--source-dataset", default="appworld")
    parser.add_argument("--source-revision", default="unknown")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite API-retriever output or report")

    train_rows = _rows(args.data)
    eval_rows = _rows(args.eval_data)
    train_ids = {str(row.meta.get("parent_record_id")) for row in train_rows}
    eval_ids = {str(row.meta.get("parent_record_id")) for row in eval_rows}
    if train_ids & eval_ids:
        raise ValueError("train/eval parent_record_id overlap")
    train_prompts, train_labels = first_action_examples(train_rows)
    eval_prompts, eval_labels = first_action_examples(eval_rows)
    argument_fields = first_action_argument_fields(train_rows)
    retriever = AppWorldAPINearestNeighbor(
        train_prompts,
        train_labels,
        dim=args.embedding_dim,
        argument_fields=argument_fields,
    )
    predictions = [retriever.predict(prompt) for prompt in eval_prompts]
    exact = sum(actual == predicted for actual, predicted in zip(eval_labels, predictions))
    parent_identity = _identity(args.init)
    source = {
        "dataset": args.source_dataset,
        "revision": args.source_revision,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_input": _identity(args.data),
        "eval_input": _identity(args.eval_data),
    }
    payload = {
        "kind": "localagent_appworld_api_retriever",
        "schema_version": 1,
        "embedding_dim": args.embedding_dim,
        "prompts": train_prompts,
        "labels": train_labels,
        "classes": list(retriever.classes),
        "argument_fields": {label: list(fields) for label, fields in argument_fields.items()},
        "parent_checkpoint": parent_identity,
        "source": source,
    }
    torch.save(payload, args.output)
    report = {
        "kind": "localagent_appworld_api_retriever_report",
        "schema_version": 1,
        "parent": parent_identity,
        "child": _identity(args.output),
        "source": source,
        "classes": list(retriever.classes),
        "metrics": {
            "eval_rows": len(eval_labels),
            "exact": exact,
            "accuracy": exact / max(1, len(eval_labels)),
        },
        "learned_weights": False,
        "claim_boundary": (
            "Char-ngram nearest-neighbor schema adapter over public AppWorld train prompts, "
            "evaluated on disjoint dev prompts. It has no learned model weights and is not a "
            "complete AppWorld trajectory, official leaderboard score, or external-account result."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
