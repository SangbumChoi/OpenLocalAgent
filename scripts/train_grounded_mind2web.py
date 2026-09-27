#!/usr/bin/env python3
"""Train a small grounded-pointer continuation on public Mind2Web DOM snapshots.

This is a bounded research continuation, not an official Mind2Web scorer.  It warm-starts the
backbone/tool head from a LocalAgent checkpoint, expands only the pointer argument vocabulary for
browser ``target_id``/``value`` slots, and keeps the normalized public train/eval files separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from localagent.agent.pointer_head import BROWSER_PTR_ARGS, PTR_ARGS, PointerHead, gold_span
from localagent.data.agent_synth import Sample
from localagent.data.schema import Conversation, Role
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.sft import sft
from localagent.train.stage_data import (
    build_continuation_lineage,
    tokenizer_identity,
)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load_rows(paths: list[Path]) -> list[Conversation]:
    rows = []
    for path in paths:
        rows.extend(
            Conversation.from_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not rows:
        raise ValueError("no conversations supplied")
    return rows


def _assert_disjoint(train: list[Conversation], evaluation: list[Conversation]) -> None:
    train_ids = {str(row.meta.get("parent_record_id")) for row in train}
    eval_ids = {str(row.meta.get("parent_record_id")) for row in evaluation}
    overlap = sorted((train_ids & eval_ids) - {"None"})
    if overlap:
        raise ValueError(f"train/eval parent_record_id overlap: {overlap[:5]}")
    train_slots = {
        str(value)
        for row in train
        for values in row.meta.get("slot_values", {}).values()
        for value in values
    }
    eval_slots = {
        str(value)
        for row in evaluation
        for values in row.meta.get("slot_values", {}).values()
        for value in values
    }
    if train_slots & eval_slots:
        raise ValueError("train/eval slot values overlap")


def _head_samples(rows: list[Conversation]) -> list[Sample]:
    samples: list[Sample] = []
    for row in rows:
        history: list[str] = []
        for message in row.messages:
            if message.role == Role.user:
                history.append(message.content)
            elif message.role == Role.tool:
                history.append(message.tool_response or "")
            elif message.role == Role.assistant and message.tool_calls:
                prompt = "\n".join(part for part in history if part)
                for call in message.tool_calls:
                    target = json.dumps(
                        {"name": call.name, "arguments": call.arguments},
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    samples.append(
                        Sample(
                            "mind2web_grounded",
                            "public_agent",
                            prompt,
                            "tool",
                            target,
                            call.name,
                            json.dumps(call.arguments, separators=(",", ":"), sort_keys=True),
                        )
                    )
    return samples


def _pointer_args(parent: dict[str, Any]) -> list[str]:
    """Return a parent-compatible vocabulary with the browser grounding rows appended."""

    parent_state = parent["ptr_head"]
    rows = int(parent_state["arg_emb.weight"].shape[0])
    declared = parent.get("ptr_args")
    if isinstance(declared, list) and len(declared) == rows:
        source_args = [str(name) for name in declared]
    elif rows == len(PTR_ARGS):
        source_args = list(PTR_ARGS)
    elif rows == len(BROWSER_PTR_ARGS):
        source_args = list(BROWSER_PTR_ARGS)
    else:
        raise ValueError(f"cannot infer parent pointer vocabulary for {rows} rows")
    result = list(source_args)
    for name in BROWSER_PTR_ARGS:
        if name not in result:
            result.append(name)
    return result


def _warm_pointer(
    parent: dict[str, Any], d_model: int, pointer_args: list[str]
) -> dict[str, torch.Tensor]:
    """Migrate any compatible parent pointer vocabulary by argument name."""

    parent_state = parent["ptr_head"]
    rows = int(parent_state["arg_emb.weight"].shape[0])
    declared = parent.get("ptr_args")
    if isinstance(declared, list) and len(declared) == rows:
        source_args = [str(name) for name in declared]
    elif rows == len(PTR_ARGS):
        source_args = list(PTR_ARGS)
    elif rows == len(BROWSER_PTR_ARGS):
        source_args = list(BROWSER_PTR_ARGS)
    else:
        raise ValueError(f"cannot infer parent pointer vocabulary for {rows} rows")
    target = PointerHead(d_model, args=pointer_args)
    target_state = target.state_dict()
    target_state["start.weight"] = parent_state["start.weight"]
    target_state["end.weight"] = parent_state["end.weight"]
    for source_index, name in enumerate(source_args):
        if name in target.arg_idx:
            target_state["arg_emb.weight"][target.arg_idx[name]] = parent_state["arg_emb.weight"][source_index]
    target.load_state_dict(target_state)
    return target.state_dict()


def _pointer_metrics(
    model: LocalAgentLM,
    ptr: PointerHead,
    tokenizer: Any,
    rows: list[Sample],
) -> dict[str, Any]:
    correct = 0
    total = 0
    for sample in rows:
        if not sample.ref_args:
            continue
        arguments = json.loads(sample.ref_args)
        pointer_arg = next(
            (name for name, value in arguments.items() if name in ptr.arg_idx and isinstance(value, str)),
            None,
        )
        if pointer_arg is None:
            continue
        value = str(arguments[pointer_arg])
        ids = tokenizer.encode(f"{USER}{sample.prompt}{ASSISTANT}")
        value_ids = tokenizer.encode(value)
        # Match inference/model context semantics.  Grounded DOM snapshots can make a single
        # history longer than the model's RoPE table; left truncation preserves the latest action
        # context and must happen before computing the gold span.
        ids = ids[-model.cfg.max_seq_len :]
        gold = gold_span(ids, value_ids)
        if gold is None:
            continue
        with torch.no_grad():
            _, hidden = model(torch.tensor([ids]), return_hidden=True)
        start, end = ptr.predict_span(hidden[0, : len(ids)], pointer_arg)
        correct += int((start, end) == gold)
        total += 1
    return {"exact_span": correct / total if total else 0.0, "span_rows": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite outputs")
    train_rows = _load_rows(args.train)
    eval_rows = _load_rows(args.eval)
    _assert_disjoint(train_rows, eval_rows)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**parent["cfg"])
    cfg.assert_within_budget()
    model = LocalAgentLM(cfg)
    model.load_state_dict(parent["state_dict"])
    tokenizer_meta = parent.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(tokenizer_meta.get("kind", "byte"), tokenizer_meta.get("path"))
    train_samples = _head_samples(train_rows)
    eval_samples = _head_samples(eval_rows)
    pointer_args = _pointer_args(parent)
    parent_identity = _identity(args.init)
    tokenizer_path = tokenizer_meta.get("path")
    tokenizer_lineage = tokenizer_identity(
        tokenizer_meta.get("kind", "byte"),
        vocab_size=cfg.vocab_size,
        path=tokenizer_path if tokenizer_path is not None else None,
    )
    # Preserve stateful/productivity rows from current parents and add browser target/value rows.
    warm_ptr = PointerHead(cfg.d_model, args=pointer_args)
    warm_ptr.load_state_dict(_warm_pointer(parent, cfg.d_model, pointer_args))
    eval_pointer_before = _pointer_metrics(model, warm_ptr, tokenizer, eval_samples)
    _, tool_head, ptr_head, metrics = sft(
        model,
        train_samples,
        tokenizer,
        conversations=train_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=max(1, min(8, args.steps // 4)),
        device=args.device,
        max_seq_len=min(1024, cfg.max_seq_len),
        joint_tool_head=True,
        ptr_args=pointer_args,
        init_tool_head=parent.get("tool_head"),
        init_ptr_head=_warm_pointer(parent, cfg.d_model, pointer_args),
        ptr_weight=1.0,
        log=print,
        return_metrics=True,
    )
    assert tool_head is not None and ptr_head is not None
    after = _pointer_metrics(model, ptr_head, tokenizer, eval_samples)
    lineage = build_continuation_lineage(
        parent=parent,
        parent_checkpoint_sha256=parent_identity["sha256"],
        config={
            "stage": "sft_grounded_mind2web",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": args.device,
            "max_seq_len": min(1024, cfg.max_seq_len),
            "pointer_args": pointer_args,
        },
        model_config=cfg.__dict__,
        data_identity={
            "train_inputs": [_identity(path) for path in args.train],
            "eval_inputs": [_identity(path) for path in args.eval],
            "train_conversations": len(train_rows),
            "eval_conversations": len(eval_rows),
            "train_decisions": len(train_samples),
            "eval_decisions": len(eval_samples),
        },
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
    )
    child = {
        "cfg": cfg.__dict__,
        "state_dict": model.state_dict(),
        "tool_head": tool_head.state_dict(),
        "ptr_head": ptr_head.state_dict(),
        "ptr_args": pointer_args,
        "route_head": parent.get("route_head"),
        "dense_selector": parent.get("dense_selector"),
        "selector_proj": parent.get("selector_proj"),
        "examples": parent.get("examples", {}),
        "tokenizer": tokenizer_meta,
        "stage": "sft_grounded_mind2web",
        "step": args.steps,
        "data": {"train": [_identity(path) for path in args.train], "eval": [_identity(path) for path in args.eval]},
        "parent_checkpoint_sha256": parent_identity["sha256"],
        "lineage": lineage,
        "training_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_grounded_mind2web_continuation_report",
        "schema_version": 1,
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "train_inputs": [_identity(path) for path in args.train],
        "eval_inputs": [_identity(path) for path in args.eval],
        "rows": {"train_conversations": len(train_rows), "eval_conversations": len(eval_rows), "train_decisions": len(train_samples), "eval_decisions": len(eval_samples)},
        "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr, "device": args.device},
        "pointer_args": pointer_args,
        "before": eval_pointer_before,
        "after": after,
        "claim_boundary": "Public Mind2Web train-record-disjoint DOM-enriched continuation; no official Mind2Web test score, emulator/browser task success, or external-account claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
