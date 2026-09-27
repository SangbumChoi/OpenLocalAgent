#!/usr/bin/env python3
"""Train a separate AgentNet selector while preserving the browser selector.

The public AgentNet projection keeps low-level desktop actions (click, drag, keyboard, and wait)
as a distinct ``agentnet_*`` tool surface.  A browser checkpoint can have a strong generic selector
and still fail here because its tool names and candidate descriptions differ.  This runner freezes
the language-model backbone, trains a matched warm/random AgentNet selector, and stores it as a
surface-specific arm alongside the browser ``dense_selector``.  It is an offline text projection:
no screenshots, desktop runtime, or OS side effects are used.
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

from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings
from localagent.data.agentnet import _tool
from localagent.data.schema import Conversation, ToolSpec
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer


ACTION_TO_TOOL = {
    "click": "agentnet_click",
    "doubleClick": "agentnet_double_click",
    "rightClick": "agentnet_right_click",
    "middleClick": "agentnet_middle_click",
    "moveTo": "agentnet_move_cursor",
    "dragTo": "agentnet_drag",
    "scroll": "agentnet_scroll",
    "hscroll": "agentnet_hscroll",
    "write": "agentnet_type_text",
    "press": "agentnet_key_press",
    "hotkey": "agentnet_hotkey",
    "tripleClick": "agentnet_triple_click",
    "wait": "agentnet_wait",
    "terminate": "agentnet_terminate",
}


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load_tokenizer(checkpoint: dict[str, Any]):
    metadata = checkpoint.get("tokenizer") or {"kind": "byte"}
    kind = str(metadata.get("kind", "byte"))
    path = metadata.get("path")
    tokenizer = load_tokenizer(kind, path)
    cfg = checkpoint["cfg"]
    vocab_size = cfg["vocab_size"] if isinstance(cfg, dict) else cfg.vocab_size
    if tokenizer.vocab_size != vocab_size:
        raise ValueError("checkpoint tokenizer vocabulary does not match model config")
    return tokenizer


def _load_rows(path: Path) -> list[Conversation]:
    rows = [
        Conversation.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty AgentNet projection: {path}")
    return rows


def _action(row: Conversation) -> str:
    code = str(row.meta.get("action_code", ""))
    for source, action in (
        ("tripleClick", "tripleClick"),
        ("rightClick", "rightClick"),
        ("middleClick", "middleClick"),
        ("doubleClick", "doubleClick"),
        ("hotkey", "hotkey"),
        ("press", "press"),
    ):
        if source in code:
            return action
    return {"click": "click", "double_click": "doubleClick", "drag": "dragTo",
            "key_press": "press", "move_cursor": "moveTo", "scroll": "scroll",
            "type_text": "write", "wait": "wait"}.get(
        row.messages[1].tool_calls[0].name, row.messages[1].tool_calls[0].name
    )


def _decision_rows(rows: list[Conversation]) -> list[tuple[str, str]]:
    decisions: list[tuple[str, str]] = []
    for row in rows:
        if not row.messages[1].tool_calls:
            continue
        tool = ACTION_TO_TOOL.get(_action(row))
        if tool is not None:
            decisions.append((row.messages[0].content, tool))
    if not decisions:
        raise ValueError("AgentNet projection contains no supported tool decisions")
    return decisions


def _features(model: LocalAgentLM, tokenizer: Any, prompts: list[str], batch_size: int = 32) -> torch.Tensor:
    encoded = [tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")[-model.cfg.max_seq_len :] for prompt in prompts]
    output: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            rows = encoded[start : start + batch_size]
            width = max(len(row) for row in rows)
            ids = torch.zeros((len(rows), width), dtype=torch.long)
            for index, row in enumerate(rows):
                ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            _, hidden = model(ids, return_hidden=True)
            output.extend(hidden[index, len(row) - 1].detach() for index, row in enumerate(rows))
    return torch.stack(output)


def _specs() -> list[ToolSpec]:
    return [
        ToolSpec(name=raw["name"], description=raw["description"], parameters=raw["parameters"])
        for action in (
            "click", "doubleClick", "rightClick", "middleClick", "moveTo", "dragTo", "scroll",
            "hscroll", "write", "press", "hotkey", "tripleClick", "wait", "terminate",
        )
        for raw in [_tool(action)]
    ]


def _train(
    features: torch.Tensor,
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    parent_state: dict[str, Any],
    *,
    warm: bool,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> DenseToolSelector:
    torch.manual_seed(seed)
    selector = DenseToolSelector(features.shape[1], emb_dim=embeddings.shape[1], proj=256)
    if warm:
        selector.load_state_dict(parent_state)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr)
    rng = random.Random(seed)
    selector.train()
    for _ in range(steps):
        indices = [rng.randrange(len(labels)) for _ in range(min(batch_size, len(labels)))]
        index = torch.tensor(indices, dtype=torch.long)
        loss = F.cross_entropy(selector(features[index], embeddings), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return selector.eval()


def _score(selector: DenseToolSelector, features: torch.Tensor, labels: torch.Tensor, embeddings: torch.Tensor) -> float:
    with torch.no_grad():
        return float((selector(features, embeddings).argmax(-1) == labels).float().mean())


def _movement(before: dict[str, Any], after: dict[str, Any]) -> float:
    numerator = sum(float((after[name].float() - value.float()).pow(2).sum()) for name, value in before.items())
    denominator = sum(float(value.float().pow(2).sum()) for value in before.values())
    return (numerator**0.5) / max(denominator**0.5, 1e-12)


def _save(parent: dict[str, Any], path: Path, selector: DenseToolSelector, examples: dict[str, list[str]], arm: str) -> None:
    child = dict(parent)
    surfaces = dict(parent.get("surface_selectors", {}))
    surfaces["agentnet"] = {
        "state_dict": selector.state_dict(),
        "selector_proj": 256,
        "examples": examples,
        "arm": arm,
    }
    child["surface_selectors"] = surfaces
    child["stage"] = "sft_agentnet_surface_selector_repair"
    child["agentnet_selector_repair"] = {"arm": arm, "backbone_frozen": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=2042)
    args = parser.parse_args()
    for path in (args.output, args.random_output, args.report):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config).eval()
    model.load_state_dict(parent["state_dict"])
    tokenizer = _load_tokenizer(parent)
    train_rows = _decision_rows(_load_rows(args.train))
    eval_rows = _decision_rows(_load_rows(args.eval))
    specs = _specs()
    name_index = {spec.name: index for index, spec in enumerate(specs)}
    train_features = _features(model, tokenizer, [prompt for prompt, _ in train_rows])
    eval_features = _features(model, tokenizer, [prompt for prompt, _ in eval_rows])
    train_labels = torch.tensor([name_index[name] for _, name in train_rows], dtype=torch.long)
    eval_labels = torch.tensor([name_index[name] for _, name in eval_rows], dtype=torch.long)
    examples: dict[str, list[str]] = {name: [] for name in name_index}
    for prompt, name in train_rows:
        if len(examples[name]) < 64:
            examples[name].append(prompt)
    embeddings = tool_embeddings(specs, examples=examples)
    parent_state = parent["dense_selector"]
    warm = _train(
        train_features, train_labels, embeddings, parent_state, warm=True, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    random_arm = _train(
        train_features, train_labels, embeddings, parent_state, warm=False, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    report = {
        "kind": "localagent_agentnet_surface_selector_repair",
        "schema_version": 1,
        "parent": _sha256(args.init),
        "train": _sha256(args.train),
        "eval": _sha256(args.eval),
        "rows": {"train": len(train_rows), "eval": len(eval_rows), "tool_surface": len(specs)},
        "warm": {"train_top1": _score(warm, train_features, train_labels, embeddings),
                 "eval_top1": _score(warm, eval_features, eval_labels, embeddings),
                 "relative_l2": _movement(parent_state, warm.state_dict())},
        "random": {"train_top1": _score(random_arm, train_features, train_labels, embeddings),
                   "eval_top1": _score(random_arm, eval_features, eval_labels, embeddings),
                   "relative_l2": _movement(parent_state, random_arm.state_dict())},
        "backbone_frozen": True,
        "claim_boundary": "AgentNet text projection selector only; no screenshots, desktop runtime, official split, or OS side effects.",
    }
    _save(parent, args.output, warm, examples, "warm")
    _save(parent, args.random_output, random_arm, examples, "random")
    report["warm_checkpoint"] = _sha256(args.output)
    report["random_checkpoint"] = _sha256(args.random_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
