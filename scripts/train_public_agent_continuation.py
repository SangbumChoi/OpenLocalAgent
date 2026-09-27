#!/usr/bin/env python3
"""Train and evaluate a held-out public-agent continuation.

The input files must already be normalized ``Conversation`` JSONL.  The script enforces
source-record-disjoint train/eval rows when ``parent_record_id`` metadata is available, reports
teacher-forced metrics before and after SFT, and binds every local input by SHA-256.  It is a
text-first public-data experiment: it does not claim emulator, browser, or external-account task
success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, train_dense_selector
from localagent.agent.routes import ROUTES, RouteHead, route_of, train_route_head
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import REALISTIC_BROWSER_TOOLS, STANDARD_TOOLS
from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import _evaluate_conversations, sft
from localagent.train.stage_data import (
    build_continuation_lineage,
    probe_decisions,
    tokenizer_identity,
)
from scripts.analyze_weight_transfer import analyze as analyze_weight_transfer


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load_rows(paths: Iterable[Path]) -> list[Conversation]:
    rows: list[Conversation] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(Conversation.from_json(line))
                except Exception as error:  # pragma: no cover - diagnostic context
                    raise ValueError(f"invalid Conversation at {path}:{line_number}") from error
    if not rows:
        raise ValueError("no normalized conversations found")
    return rows


def _assert_disjoint(train: list[Conversation], evaluation: list[Conversation]) -> None:
    train_records = {
        str(row.meta["parent_record_id"])
        for row in train
        if isinstance(row.meta, dict) and row.meta.get("parent_record_id")
    }
    eval_records = {
        str(row.meta["parent_record_id"])
        for row in evaluation
        if isinstance(row.meta, dict) and row.meta.get("parent_record_id")
    }
    if train_records and eval_records and train_records & eval_records:
        overlap = sorted(train_records & eval_records)
        raise ValueError(f"train/eval parent_record_id overlap: {overlap[:5]}")

    def slots(rows: list[Conversation]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for row in rows:
            for name, values in row.meta.get("slot_values", {}).items():
                result.setdefault(str(name), set()).update(str(value) for value in values)
        return result

    train_slots, eval_slots = slots(train), slots(evaluation)
    overlaps = {
        name: sorted(train_slots.get(name, set()) & eval_slots.get(name, set()))
        for name in set(train_slots) | set(eval_slots)
    }
    overlaps = {name: values for name, values in overlaps.items() if values}
    if overlaps:
        raise ValueError(f"train/eval slot overlap: {overlaps}")


def _checkpoint_tokenizer(parent: dict[str, Any]):
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


def _load_heads(
    parent: dict[str, Any],
    model: LocalAgentLM,
    *,
    selector_tools,
    init: str = "parent",
    seed: int = 2027,
) -> tuple[RouteHead, BoundSelector]:
    if init not in {"parent", "random"}:
        raise ValueError(f"unsupported head initialization: {init!r}")
    selector_state = parent["dense_selector"]
    # Keep random-head initialization from perturbing the subsequent deterministic SFT sampling
    # stream.  This makes the parent-vs-random head comparison use identical backbone updates.
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    route = RouteHead(model.cfg.d_model)
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=selector_state["t_proj.weight"].shape[1],
        proj=selector_state["q_proj.weight"].shape[0],
    )
    torch.set_rng_state(rng_state)
    if init == "parent":
        route.load_state_dict(parent["route_head"])
        selector.load_state_dict(selector_state)
    route.eval()
    selector.eval()
    return route, BoundSelector(selector, selector_tools, examples=parent.get("examples", {}))


def _head_metrics(
    model: LocalAgentLM,
    tokenizer: Any,
    route: RouteHead,
    selector: BoundSelector,
    rows: list[Conversation],
) -> dict[str, Any]:
    decisions = probe_decisions(rows)
    if not decisions:
        return {"rows": 0, "tool_rows": 0, "route_accuracy": 0.0, "selector_top1_accuracy": 0.0}
    features = torch.stack(
        [
            _feat(model, tokenizer, decision.prompt, "cpu", framed=decision.framed)
            for decision in decisions
        ]
    )
    with torch.no_grad():
        route_predictions = route(features).argmax(-1).tolist()
        selector_predictions = selector.model(features, selector.embs).argmax(-1).tolist()
    route_correct = 0
    selector_correct = 0
    tool_rows = 0
    for index, decision in enumerate(decisions):
        route_correct += int(ROUTES[route_predictions[index]] == route_of(decision.ref_name))
        if decision.kind == "tool" and decision.ref_name in selector.names:
            tool_rows += 1
            selector_correct += int(selector.names[selector_predictions[index]] == decision.ref_name)
    return {
        "rows": len(decisions),
        "tool_rows": tool_rows,
        "route_accuracy": route_correct / len(decisions),
        "selector_top1_accuracy": selector_correct / tool_rows if tool_rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--eval-data", type=Path, action="append", required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-dataset", default="unknown")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--head-steps", type=int, default=0)
    parser.add_argument("--head-init", choices=("parent", "random"), default="parent")
    parser.add_argument(
        "--selector-pool",
        choices=("realistic", "standard"),
        default="realistic",
        help="tool pool used to train/bind the dense selector; keep it identical to runtime tools",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite continuation outputs")
    if args.steps < 1 or args.head_steps < 0 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps and batch-size must be positive; head-steps cannot be negative")

    train_rows = _load_rows(args.data)
    eval_rows = _load_rows(args.eval_data)
    _assert_disjoint(train_rows, eval_rows)
    parent_identity = _identity(args.init)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)
    tokenizer_meta = parent.get("tokenizer") or {"kind": "byte"}
    tokenizer_path = tokenizer_meta.get("path")
    tokenizer_lineage = tokenizer_identity(
        tokenizer_meta.get("kind", "byte"),
        vocab_size=config.vocab_size,
        path=tokenizer_path if tokenizer_path is not None else None,
    )
    selector_tools = STANDARD_TOOLS if args.selector_pool == "standard" else REALISTIC_BROWSER_TOOLS
    route, selector = _load_heads(
        parent,
        model,
        selector_tools=selector_tools,
        init=args.head_init,
        seed=2027,
    )
    heads_before = _head_metrics(model, tokenizer, route, selector, eval_rows)

    before_train = _evaluate_conversations(
        model, train_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    before_eval = _evaluate_conversations(
        model, eval_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    loss_history, _, _, training = sft(
        model,
        [],
        tokenizer,
        conversations=train_rows,
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
    after_train = _evaluate_conversations(
        model, train_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    after_eval = _evaluate_conversations(
        model, eval_rows, tokenizer, max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device
    )
    decisions = probe_decisions(train_rows)
    if args.head_steps:
        route = train_route_head(
            model,
            decisions,
            tokenizer,
            steps=args.head_steps,
            batch_size=128,
            device=args.device,
            log=print,
        )
        selector_model = train_dense_selector(
            model,
            decisions,
            tokenizer,
            selector_tools,
            steps=args.head_steps,
            batch_size=128,
            device=args.device,
            proj=int(parent["selector_proj"]),
            examples=parent.get("examples", {}),
            log=print,
        )
        selector = BoundSelector(
            selector_model,
            selector_tools,
            examples=parent.get("examples", {}),
        )
    heads_after = _head_metrics(model, tokenizer, route, selector, eval_rows)

    lineage = build_continuation_lineage(
        parent=parent,
        parent_checkpoint_sha256=parent_identity["sha256"],
        config={
            "stage": "sft_public_agent_continuation",
            "source_dataset": args.source_dataset,
            "source_revision": args.source_revision,
            "steps": args.steps,
            "head_steps": args.head_steps,
            "head_init": args.head_init,
            "selector_pool": args.selector_pool,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "seed": 2027,
        },
        model_config=config.__dict__,
        data_identity={
            "train_inputs": [_identity(path) for path in args.data],
            "eval_inputs": [_identity(path) for path in args.eval_data],
            "source_manifest": (
                _identity(args.source_manifest) if args.source_manifest else None
            ),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
    )

    child = dict(parent)
    child.update(
        {
            "state_dict": model.state_dict(),
            "route_head": route.state_dict(),
            "dense_selector": selector.model.state_dict(),
            "stage": "sft_public_agent_continuation",
            "parent_checkpoint_sha256": parent_identity["sha256"],
            "lineage": lineage,
            "steps": args.steps,
            "public_agent_training": {
                "dataset": args.source_dataset,
                "revision": args.source_revision,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "head_steps": args.head_steps,
                "head_init": args.head_init,
                "selector_pool": args.selector_pool,
                "train_inputs": [_identity(path) for path in args.data],
                "eval_inputs": [_identity(path) for path in args.eval_data],
                "before_train": before_train,
                "after_train": after_train,
                "before_eval": before_eval,
                "after_eval": after_eval,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    transfer = analyze_weight_transfer(args.init, args.output)
    report = {
        "kind": "localagent_public_agent_continuation_report",
        "schema_version": 1,
        "source": {
            "dataset": args.source_dataset,
            "revision": args.source_revision,
            "manifest": _identity(args.source_manifest) if args.source_manifest else None,
        },
        "parent": parent_identity,
        "child": _identity(args.output),
        "train_inputs": [_identity(path) for path in args.data],
        "eval_inputs": [_identity(path) for path in args.eval_data],
        "rows": {"train": len(train_rows), "eval": len(eval_rows)},
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "head_steps": args.head_steps,
            "head_init": args.head_init,
            "selector_pool": args.selector_pool,
            "max_seq_len": args.max_seq_len,
            "seed": 2027,
            "device": args.device,
        },
        "before": {"train": before_train, "eval": before_eval},
        "after": {"train": after_train, "eval": after_eval},
        "heads": {"before": heads_before, "after": heads_after},
        "loss_history": loss_history,
        "token_accounting": training,
        "weight_transfer": {
            "base": transfer["base"],
            "target": transfer["target"],
            "compatibility": transfer["compatibility"],
            "groups": transfer["groups"],
            "recommendation": transfer["recommendation"],
        },
        "claim_boundary": (
            f"Held-out public {args.source_dataset} source continuation with text-first "
            "teacher-forced metrics; this is not an official benchmark score and makes no "
            "native browser/emulator/MCP/external-account claim."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
