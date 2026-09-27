#!/usr/bin/env python
"""Run a bounded text-first SFT continuation on normalized AndroidControl rows.

This is intentionally a small, reproducible bridge experiment: the official AndroidControl
TFRecords are decoded offline, accessibility trees are projected to text, and the resulting
Conversation JSONL is used as multi-turn LM supervision.  It does not claim Android task success
because no emulator is launched here; the report records loss/accuracy on the normalized rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import _evaluate_conversations, sft


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _load_rows(path: Path) -> list[Conversation]:
    rows: list[Conversation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(Conversation.from_json(line))
                except Exception as error:  # pragma: no cover - diagnostic context
                    raise ValueError(f"invalid Conversation at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"no conversations found in {path}")
    return rows


def _checkpoint_tokenizer(parent: dict):
    """Load the tokenizer recorded by the parent checkpoint, never defaulting silently to bytes."""

    metadata = parent.get("tokenizer") or {"kind": "byte"}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")
    kind = str(metadata.get("kind", "byte"))
    path = metadata.get("path")
    if kind == "bpe" and path is None:
        raise ValueError("BPE checkpoint is missing tokenizer.path")
    tokenizer = load_tokenizer(kind, path)
    cfg = parent.get("cfg")
    vocab_size = cfg.get("vocab_size") if isinstance(cfg, dict) else getattr(cfg, "vocab_size", None)
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary "
            f"({tokenizer.vocab_size}) does not match model config ({vocab_size})"
        )
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="normalized Conversation JSONL; repeat to train on a verified mixture",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        action="append",
        default=[],
        help="held-out normalized Conversation JSONL; never used for optimization",
    )
    parser.add_argument("--init", type=Path, required=True, help="parent SFT checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="child checkpoint path")
    parser.add_argument("--report", type=Path, required=True, help="JSON metrics report")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")

    rows = [row for path in args.data for row in _load_rows(path)]
    eval_rows = [row for path in args.eval_data for row in _load_rows(path)]
    if not rows:
        raise ValueError("no training rows found")
    data_identities = [
        {"path": str(path), "bytes": _sha256(path)[0], "sha256": _sha256(path)[1]}
        for path in args.data
    ]
    eval_identities = [
        {"path": str(path), "bytes": _sha256(path)[0], "sha256": _sha256(path)[1]}
        for path in args.eval_data
    ]
    visual_omitted_rows = sum(bool(row.meta.get("visual_input_omitted")) for row in rows)
    eval_visual_omitted_rows = sum(bool(row.meta.get("visual_input_omitted")) for row in eval_rows)
    parent_bytes, parent_sha = _sha256(args.init)
    parent = torch.load(args.init, map_location="cpu")
    cfg = ModelConfig(**parent["cfg"])
    model = LocalAgentLM(cfg)
    model.load_state_dict(parent["state_dict"])
    tok = _checkpoint_tokenizer(parent)

    before = _evaluate_conversations(
        model,
        rows,
        tok,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    before_eval = (
        _evaluate_conversations(
            model,
            eval_rows,
            tok,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )
        if eval_rows
        else None
    )
    loss_history, _, _, training = sft(
        model,
        [],
        tok,
        conversations=rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=max(1, min(4, args.steps // 2)),
        device=args.device,
        max_seq_len=args.max_seq_len,
        seed=2027,
        log=print,
        return_metrics=True,
    )
    after = _evaluate_conversations(
        model,
        rows,
        tok,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    after_eval = (
        _evaluate_conversations(
            model,
            eval_rows,
            tok,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )
        if eval_rows
        else None
    )

    child = dict(parent)
    child.update(
        {
            "state_dict": model.state_dict(),
            "stage": "sft_realistic_mobile_pilot",
            "step": int(parent.get("step", 0)) + args.steps,
            "parent_checkpoint_sha256": parent_sha,
            "parent_checkpoint_bytes": parent_bytes,
            "steps": args.steps,
            "data": {
                "normalized_conversations": [str(path) for path in args.data],
                "identities": data_identities,
                "eval_conversations": [str(path) for path in args.eval_data],
                "eval_identities": eval_identities,
                "rows": len(rows),
                "eval_rows": len(eval_rows),
                "source": "google/androidcontrol",
                "visual_input_omitted_rows": visual_omitted_rows,
                "eval_visual_input_omitted_rows": eval_visual_omitted_rows,
            },
            "androidcontrol_training": {
                "max_seq_len": args.max_seq_len,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "before": before,
                "after": after,
                "before_eval": before_eval,
                "after_eval": after_eval,
                "loss_history": loss_history,
                "token_accounting": training,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_androidcontrol_pilot_training_report",
        "parent": {"path": str(args.init), "bytes": parent_bytes, "sha256": parent_sha},
        "child": {"path": str(args.output), **dict(zip(("bytes", "sha256"), _sha256(args.output)))},
        "data": {
            "identities": data_identities,
            "eval_identities": eval_identities,
            "visual_input_omitted_rows": visual_omitted_rows,
            "eval_visual_input_omitted_rows": eval_visual_omitted_rows,
        },
        "rows": len(rows),
        "eval_rows": len(eval_rows),
        "steps": args.steps,
        "before": before,
        "after": after,
        "before_eval": before_eval,
        "after_eval": after_eval,
        "loss_history": loss_history,
        "training": training,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
