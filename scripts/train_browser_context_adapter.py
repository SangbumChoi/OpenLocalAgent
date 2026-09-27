#!/usr/bin/env python3
"""Adapt a checkpoint to live accessibility-tree browser prompts.

This is a bounded, train-only adapter.  It projects the existing synthetic GUI training rows into
the same observation shape used by the native BrowserGym runner, then trains the backbone at a low
learning rate and rebuilds the frozen-feature route/selector probes.  Evaluation rows are projected
with the same deterministic transform but are never used for optimization.

The script deliberately does not read BrowserGym task plans, reset goals, screenshots, or labels.
It is therefore a context-contract adaptation experiment, not BrowserGym training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, train_dense_selector
from localagent.agent.parser import extract_tool_calls
from localagent.agent.routes import ROUTES, RouteHead, train_route_head
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.schema import Conversation, Message, Role
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import sft
from localagent.train.stage_data import load_conversation_source, probe_decisions, single_turn_samples

COMPUTER_TOOLS = frozenset(
    {
        "click",
        "double_click",
        "type_text",
        "key_press",
        "scroll",
        "drag",
        "wait",
        "move_cursor",
        "open_app",
        "screenshot",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elements(call_name: str, arguments: dict[str, Any]) -> str:
    if call_name in {"click", "double_click"}:
        target = str(arguments.get("target", "Okay"))
        return f'[21] button: "{target}"\n[22] button: "Cancel"'
    if call_name == "type_text":
        return '[21] textbox: "Search"\n[22] button: "Submit"'
    if call_name == "key_press":
        return '[21] textbox: "Input"\n[22] button: "Submit"'
    if call_name == "scroll":
        return '[21] button: "Previous"\n[22] button: "Next"'
    if call_name == "drag":
        source = str(arguments.get("source", "Card A"))
        dest = str(arguments.get("dest", "Card B"))
        return f'[21] button: "{source}"\n[22] button: "{dest}"'
    if call_name == "move_cursor":
        return f'[21] button: "{arguments.get("target", "Menu")}"'
    if call_name == "open_app":
        return '[21] button: "Chrome"\n[22] button: "Terminal"'
    if call_name == "wait":
        return '[21] status: "Loading"'
    return '[21] document: "Page"'


def _project_prompt(prompt: str, call_name: str, arguments: dict[str, Any]) -> str:
    return (
        f"Browser task: {prompt}\n\n"
        "Live accessibility elements (quoted names are valid targets):\n"
        f"{_elements(call_name, arguments)}\n\n"
        "Choose exactly one grounded computer action or abstain."
    )


def project_conversation(conversation: Conversation) -> Conversation | None:
    """Project one single-turn computer-use row, or return ``None`` for other rows."""

    messages = conversation.messages
    if len(messages) != 2 or messages[0].role != Role.user or messages[1].role != Role.assistant:
        return None
    calls = messages[1].tool_calls
    if len(calls) != 1 or calls[0].name not in COMPUTER_TOOLS:
        return None
    call = calls[0]
    projected = copy.deepcopy(conversation)
    projected.messages[0] = Message(
        role=Role.user,
        content=_project_prompt(messages[0].content, call.name, dict(call.arguments)),
    )
    projected.meta = {
        **conversation.meta,
        "projection": "synthetic_train_only_browser_context_v1",
    }
    return projected


def project_source(path: Path, *, expected_split: str) -> tuple[list[Conversation], dict[str, int]]:
    source = load_conversation_source(path, require_verified=False, expected_split=expected_split)
    rows: list[Conversation] = []
    by_tool: dict[str, int] = {}
    for conversation in source.conversations:
        projected = project_conversation(conversation)
        if projected is None:
            continue
        name = projected.messages[1].tool_calls[0].name
        rows.append(projected)
        by_tool[name] = by_tool.get(name, 0) + 1
    return rows, dict(sorted(by_tool.items()))


def _load_checkpoint(path: Path, tokenizer_path: Path) -> tuple[dict[str, Any], LocalAgentLM, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokenizer = load_tokenizer("bpe", tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint and adapter tokenizer vocabulary sizes disagree")
    return checkpoint, model, tokenizer


def _heads(checkpoint: dict[str, Any], model: LocalAgentLM) -> tuple[RouteHead, BoundSelector]:
    route = RouteHead(model.cfg.d_model)
    route.load_state_dict(checkpoint["route_head"])
    route.eval()
    selector_state = checkpoint["dense_selector"]
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=selector_state["t_proj.weight"].shape[1],
        proj=selector_state["q_proj.weight"].shape[0],
    )
    selector.load_state_dict(selector_state)
    selector.eval()
    return route, BoundSelector(selector, STANDARD_TOOLS, examples=checkpoint.get("examples", {}))


def _score(
    model: LocalAgentLM,
    tokenizer: Any,
    route: RouteHead,
    selector: BoundSelector,
    rows: list[Conversation],
) -> dict[str, Any]:
    route_correct = tool_correct = argument_correct = 0
    records: list[dict[str, Any]] = []
    for conversation in rows:
        prompt = conversation.messages[0].content
        gold = conversation.messages[1].tool_calls[0]
        output = hybrid_decode(
            model,
            tokenizer,
            prompt,
            STANDARD_TOOLS,
            selector=selector,
            route_head=route,
            top_m=1,
        )
        calls = extract_tool_calls(output)
        predicted = calls[0] if calls else None
        route_ok = ROUTES[int(route(_last_feature(model, tokenizer, prompt)).argmax(-1))] == "computer_use"
        tool_ok = predicted is not None and predicted.name == gold.name
        args_ok = tool_ok and predicted.arguments == gold.arguments
        route_correct += int(route_ok)
        tool_correct += int(tool_ok)
        argument_correct += int(args_ok)
        records.append(
            {
                "gold_tool": gold.name,
                "predicted_tool": predicted.name if predicted else None,
                "route_correct": route_ok,
                "tool_exact": tool_ok,
                "arguments_exact": args_ok,
            }
        )
    total = len(rows)
    return {
        "rows": total,
        "route_accuracy": route_correct / total if total else 0.0,
        "tool_accuracy": tool_correct / total if total else 0.0,
        "argument_accuracy": argument_correct / total if total else 0.0,
        "records": records,
    }


def _last_feature(model: LocalAgentLM, tokenizer: Any, prompt: str) -> torch.Tensor:
    from localagent.agent.tool_head import _feat

    return _feat(model, tokenizer, prompt, "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--head-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite adapter outputs")
    if args.steps < 1 or args.head_steps < 1 or args.lr <= 0:
        raise SystemExit("steps, head-steps, and lr must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_rows, train_tools = project_source(args.train, expected_split="train")
    eval_rows, eval_tools = project_source(args.eval, expected_split="eval")
    if not train_rows or not eval_rows:
        raise SystemExit("both train and eval sources need projected computer-use rows")
    checkpoint, model, tokenizer = _load_checkpoint(args.checkpoint, args.tokenizer)
    route_before, selector_before = _heads(checkpoint, model)
    before = _score(model, tokenizer, route_before, selector_before, eval_rows)
    samples = single_turn_samples(train_rows)
    matmul_precision = torch.get_float32_matmul_precision()
    history, _, _, metrics = sft(
        model,
        samples,
        tokenizer,
        steps=args.steps,
        batch_size=4,
        accum_steps=4,
        lr=args.lr,
        warmup=max(1, args.steps // 10),
        device="cpu",
        joint_tool_head=False,
        max_seq_len=model.cfg.max_seq_len,
        seed=args.seed,
        return_metrics=True,
        log=lambda message: print(message, flush=True),
    )
    model.eval()
    decisions = probe_decisions(train_rows)
    route = train_route_head(model, decisions, tokenizer, steps=args.head_steps, batch_size=128)
    selector = train_dense_selector(
        model,
        decisions,
        tokenizer,
        STANDARD_TOOLS,
        steps=args.head_steps,
        batch_size=128,
        proj=int(checkpoint["selector_proj"]),
        examples=checkpoint.get("examples", {}),
    )
    route.eval()
    bound = BoundSelector(selector, STANDARD_TOOLS, examples=checkpoint.get("examples", {}))
    after = _score(model, tokenizer, route, bound, eval_rows)
    payload = dict(checkpoint)
    payload.update(
        {
            "state_dict": model.state_dict(),
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "stage": "sft_browser_context",
            "training": {
                "kind": "synthetic_train_only_browser_context_v1",
                "seed": args.seed,
                "steps": args.steps,
                "head_steps": args.head_steps,
                "lr": args.lr,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_source_sha256": _sha256(args.train),
                "eval_source_sha256": _sha256(args.eval),
                "loss_first": history[0],
                "loss_last": history[-1],
                "metrics": metrics,
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report = {
        "kind": "localagent_browser_context_adapter_report",
        "schema_version": 1,
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "output": {"path": str(args.out), "sha256": _sha256(args.out)},
        "tokenizer": {"path": str(args.tokenizer), "sha256": _sha256(args.tokenizer)},
        "train": {"path": str(args.train), "sha256": _sha256(args.train), "rows": len(train_rows), "tools": train_tools},
        "eval": {"path": str(args.eval), "sha256": _sha256(args.eval), "rows": len(eval_rows), "tools": eval_tools},
        "before": before,
        "after": after,
        "claim_boundary": "Train-only synthetic observation-contract adaptation; not a BrowserGym score and not training on BrowserGym task prompts or labels.",
        "matmul_precision": matmul_precision,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"before": before, "after": after, "output": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
