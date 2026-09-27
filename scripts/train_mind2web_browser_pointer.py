#!/usr/bin/env python3
"""Continue a checkpoint on Mind2Web browser actions mapped to deployed standard tools.

Mind2Web uses ``web_click(target_id)``, ``web_type(target_id,text)``, and ``web_select``.  The
deployed BrowserGym bridge instead consumes standard ``click(target)`` and ``type_text(text)``.
This adapter preserves the public DOM/task prompts, maps only the action schema, expands the
pointer vocabulary with ``target`` and ``text``, and never reads BrowserGym task labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from localagent.agent.pointer_head import PTR_ARGS, PointerHead
from localagent.data.schema import Conversation, Role, ToolCall
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import sft
from localagent.train.stage_data import canonical_sha256
from scripts.train_grounded_mind2web import _assert_disjoint, _head_samples, _pointer_metrics


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> list[Conversation]:
    rows = [Conversation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty Mind2Web source: {path}")
    return rows


def _standardize(rows: list[Conversation]) -> list[Conversation]:
    result = copy.deepcopy(rows)
    for row in result:
        for message in row.messages:
            if message.role != Role.assistant:
                continue
            calls: list[ToolCall] = []
            for call in message.tool_calls:
                if call.name == "web_click":
                    calls.append(ToolCall("click", {"target": str(call.arguments.get("target_id", ""))}))
                elif call.name == "web_type":
                    calls.append(ToolCall("type_text", {"text": str(call.arguments.get("text", ""))}))
                elif call.name == "web_select":
                    calls.append(ToolCall("click", {"target": str(call.arguments.get("target_id", ""))}))
                else:
                    raise ValueError(f"unsupported Mind2Web action: {call.name}")
            message.tool_calls = calls
        row.meta = {**row.meta, "projection": "mind2web_standard_browser_pointer_v1"}
    return result


def _pointer_args(parent: dict[str, Any]) -> list[str]:
    state = parent["ptr_head"]
    rows = int(state["arg_emb.weight"].shape[0])
    declared = parent.get("ptr_args")
    if isinstance(declared, list) and len(declared) == rows:
        source = [str(name) for name in declared]
    elif rows == len(PTR_ARGS):
        # The current WebGPU parent predates explicit ptr_args metadata.  Infer only the
        # immutable canonical 17-row vocabulary; unknown widths must still fail closed.
        source = list(PTR_ARGS)
    else:
        raise ValueError(f"cannot infer parent pointer vocabulary for {rows} rows")
    result = list(source)
    for name in ("target", "text"):
        if name not in result:
            result.append(name)
    return result


def _warm_pointer(parent: dict[str, Any], pointer_args: list[str]) -> dict[str, torch.Tensor]:
    source = parent["ptr_head"]
    rows = int(source["arg_emb.weight"].shape[0])
    declared = parent.get("ptr_args")
    if isinstance(declared, list) and len(declared) == rows:
        source_args = [str(name) for name in declared]
    elif rows == len(PTR_ARGS):
        source_args = list(PTR_ARGS)
    else:
        raise ValueError(f"cannot infer parent pointer vocabulary for {rows} rows")
    target = PointerHead(parent["cfg"]["d_model"], args=pointer_args)
    state = target.state_dict()
    state["start.weight"] = source["start.weight"]
    state["end.weight"] = source["end.weight"]
    for index, name in enumerate(source_args):
        state["arg_emb.weight"][target.arg_idx[name]] = source["arg_emb.weight"][index]
    target.load_state_dict(state)
    return target.state_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2049)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite browser-pointer outputs")
    train_rows = _standardize(_load(args.train))
    eval_rows = _standardize(_load(args.eval))
    _assert_disjoint(train_rows, eval_rows)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    torch.manual_seed(args.seed)
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    tokenizer_meta = parent.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(tokenizer_meta["kind"], tokenizer_meta.get("path"))
    pointer_args = _pointer_args(parent)
    warm_pointer = _warm_pointer(parent, pointer_args)
    train_samples = _head_samples(train_rows)
    eval_samples = _head_samples(eval_rows)
    pointer_before = PointerHead(config.d_model, args=pointer_args)
    pointer_before.load_state_dict(warm_pointer)
    before = _pointer_metrics(model, pointer_before, tokenizer, eval_samples)
    _, tool_head, pointer, metrics = sft(
        model,
        train_samples,
        tokenizer,
        conversations=train_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=max(1, min(8, args.steps // 4)),
        device="cpu",
        max_seq_len=min(1024, config.max_seq_len),
        joint_tool_head=True,
        ptr_args=pointer_args,
        init_tool_head=parent.get("tool_head"),
        init_ptr_head=warm_pointer,
        ptr_weight=1.0,
        seed=args.seed,
        log=print,
        return_metrics=True,
    )
    assert tool_head is not None and pointer is not None
    after = _pointer_metrics(model, pointer, tokenizer, eval_samples)
    child = dict(parent)
    child.update({"state_dict": model.state_dict(), "tool_head": tool_head.state_dict(), "ptr_head": pointer.state_dict(), "ptr_args": pointer_args, "stage": "sft_mind2web_standard_browser_pointer"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_mind2web_standard_browser_pointer_transfer",
        "schema_version": 1,
        "source": {"dataset": "osunlp/Mind2Web", "url": "https://huggingface.co/datasets/osunlp/Mind2Web", "revision": "17ece8eb89862368edc0cc806acee6fca5163474", "train": _identity(args.train), "eval": _identity(args.eval), "train_rows": len(train_rows), "eval_rows": len(eval_rows)},
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed, "pointer_args": pointer_args, "projection": "web_click->click(target), web_type->type_text(text), web_select->click(target)"},
        "before": before,
        "after": after,
        "training_metrics": metrics,
        "decision": {"native_replay_required": True, "adoption": "pending_native_browsergym_canary", "claim_boundary": "Train-only public Mind2Web DOM/action continuation after action-schema projection; no official Mind2Web test score, BrowserGym labels, screenshots, or external side effects."},
    }
    report["receipt_self_sha256"] = canonical_sha256(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
