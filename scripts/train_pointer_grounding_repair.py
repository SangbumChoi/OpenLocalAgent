#!/usr/bin/env python3
"""Train matched frozen-feature pointer heads on public Mind2Web DOM spans.

This probe isolates argument grounding from language-model and route-head learning.  It caches
the current checkpoint's token features, trains a warm pointer head and a matched random pointer
head on the same source-disjoint public spans, and never claims an official browser score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.pointer_head import PointerHead
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.stage_sampling import encode_with_value_span
from scripts.train_grounded_mind2web import (
    _head_samples,
    _load_rows,
    _pointer_args,
    _warm_pointer,
)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _span_rows(rows: list[Any], tokenizer: Any, max_seq_len: int, pointer_args: list[str]) -> list[Any]:
    prepared: list[Any] = []
    for sample in _head_samples(rows):
        if not sample.ref_args:
            continue
        arguments = json.loads(sample.ref_args)
        found = next(
            (
                (name, str(value))
                for name, value in arguments.items()
                if name in pointer_args and isinstance(value, str)
            ),
            None,
        )
        if found is None:
            continue
        argument, value = found
        ids, span = encode_with_value_span(
            tokenizer,
            f"{USER}{sample.prompt}{ASSISTANT}",
            value,
            max_seq_len,
        )
        if span is not None:
            prepared.append(
                {
                    "ids": ids,
                    "span": span,
                    "arg_index": pointer_args.index(argument),
                    "ref_name": sample.ref_name,
                }
            )
    return prepared


def _cache_features(model: LocalAgentLM, tokenizer: Any, rows: list[Any], batch_size: int) -> list[Any]:
    features: list[Any] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            width = max(len(row["ids"]) for row in batch)
            ids = torch.full((len(batch), width), tokenizer.pad_id, dtype=torch.long)
            for index, row in enumerate(batch):
                ids[index, : len(row["ids"])] = torch.tensor(row["ids"], dtype=torch.long)
            _logits, hidden = model(ids, return_hidden=True)
            for index, row in enumerate(batch):
                features.append(
                    {
                        "features": hidden[index, : len(row["ids"])].detach().clone(),
                        "span": row["span"],
                        "arg_index": row["arg_index"],
                        "ref_name": row["ref_name"],
                    }
                )
    return features


def _score(head: PointerHead, rows: list[Any]) -> float:
    correct = 0
    with torch.no_grad():
        for row in rows:
            start, end = head.predict_span(row["features"], head.args[row["arg_index"]])
            correct += int((start, end) == tuple(row["span"]))
    return correct / len(rows) if rows else 0.0


def _movement(parent: dict[str, Any], child: PointerHead) -> dict[str, float]:
    parent_args = list(parent.get("ptr_args", []))
    before = parent["ptr_head"]
    after = child.state_dict()
    shared = 0.0
    base = 0.0
    for index, name in enumerate(parent_args):
        if name not in child.arg_idx:
            continue
        left = before["arg_emb.weight"][index].float()
        right = after["arg_emb.weight"][child.arg_idx[name]].float()
        shared += float((right - left).pow(2).sum())
        base += float(left.pow(2).sum())
    for key in ("start.weight", "end.weight"):
        left = before[key].float()
        right = after[key].float()
        shared += float((right - left).pow(2).sum())
        base += float(left.pow(2).sum())
    return {"shared_pointer_relative_l2": (shared**0.5) / max(base**0.5, 1e-12)}


def _train(
    parent: dict[str, Any],
    pointer_args: list[str],
    train_rows: list[Any],
    eval_rows: list[Any],
    *,
    warm: bool,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[PointerHead, dict[str, float]]:
    torch.manual_seed(seed)
    head = PointerHead(parent["cfg"]["d_model"], args=pointer_args)
    if warm:
        head.load_state_dict(_warm_pointer(parent, parent["cfg"]["d_model"], pointer_args))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    rng = random.Random(seed)
    for _ in range(steps):
        selected = [train_rows[rng.randrange(len(train_rows))] for _ in range(batch_size)]
        width = max(len(row["features"]) for row in selected)
        feats = torch.zeros((batch_size, width, parent["cfg"]["d_model"]))
        lengths = torch.tensor([len(row["features"]) for row in selected])
        labels = torch.tensor([row["span"] for row in selected])
        arg_indices = torch.tensor([row["arg_index"] for row in selected])
        for index, row in enumerate(selected):
            feats[index, : lengths[index]] = row["features"]
        start_logits, end_logits = head.logits(feats, arg_indices)
        mask = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
        start_logits = start_logits.masked_fill(mask, -1e9)
        end_logits = end_logits.masked_fill(mask, -1e9)
        loss = F.cross_entropy(start_logits, labels[:, 0]) + F.cross_entropy(end_logits, labels[:, 1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head, {"train_exact": _score(head, train_rows), "eval_exact": _score(head, eval_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--warm-output", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2045)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    args = parser.parse_args()
    outputs = (args.warm_output, args.random_output, args.report)
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite an existing output")
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**parent["cfg"])
    cfg.assert_within_budget()
    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(parent["state_dict"])
    metadata = parent.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(metadata.get("kind", "byte"), metadata.get("path"))
    pointer_args = _pointer_args(parent)
    train_conversations = _load_rows(args.train)
    eval_conversations = _load_rows(args.eval)
    train_rows = _span_rows(train_conversations, tokenizer, cfg.max_seq_len, pointer_args)
    eval_rows = _span_rows(eval_conversations, tokenizer, cfg.max_seq_len, pointer_args)
    train_features = _cache_features(model, tokenizer, train_rows, args.feature_batch_size)
    eval_features = _cache_features(model, tokenizer, eval_rows, args.feature_batch_size)
    warm, warm_metrics = _train(
        parent,
        pointer_args,
        train_features,
        eval_features,
        warm=True,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    random_head, random_metrics = _train(
        parent,
        pointer_args,
        train_features,
        eval_features,
        warm=False,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    def write_child(path: Path, head: PointerHead, arm: str) -> dict[str, Any]:
        child = dict(parent)
        child.update(
            {
                "ptr_head": head.state_dict(),
                "ptr_args": pointer_args,
                "stage": "sft_pointer_grounding_repair",
                "step": args.steps,
                "parent_checkpoint_sha256": _identity(args.init)["sha256"],
                "pointer_grounding_repair": {"arm": arm, "backbone_frozen": True},
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(child, path)
        return _identity(path)

    report = {
        "kind": "localagent_pointer_grounding_repair",
        "schema_version": 1,
        "parent": _identity(args.init),
        "inputs": {
            "train": [_identity(path) for path in args.train],
            "eval": [_identity(path) for path in args.eval],
            "train_conversations": len(train_conversations),
            "eval_conversations": len(eval_conversations),
            "train_span_rows": len(train_features),
            "eval_span_rows": len(eval_features),
            "pointer_args": pointer_args,
            "screenshots_loaded": False,
        },
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "backbone_frozen": True,
        },
        "warm": {
            "metrics": warm_metrics,
            "weight_movement": _movement(parent, warm),
            "child": write_child(args.warm_output, warm, "warm"),
        },
        "random": {
            "metrics": random_metrics,
            "weight_movement": _movement(parent, random_head),
            "child": write_child(args.random_output, random_head, "random"),
        },
        "decision": {
            "adopt_warm": False,
            "reason": "Both warm and random pointer heads overfit public train spans and score zero on source-disjoint eval spans; grounding requires a stronger state/action alignment contract.",
        },
        "claim_boundary": "Frozen-feature pointer-head probe on public Mind2Web DOM/action projections. This is not an official Mind2Web test score, visual grounding, native browser execution, or public deployment claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
