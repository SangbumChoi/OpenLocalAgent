#!/usr/bin/env python3
"""Adapt the pointer/copy head on the ToolACE action-history projection.

Tool selection and argument grounding are separate deployment bottlenecks.  This probe trains a
schema-agnostic pointer head only where a string argument occurs verbatim in the catalog/history
context, comparing warm-start and matched-random frozen-backbone arms.  It is intentionally
bounded and diagnostic; it does not execute ToolACE or claim an official benchmark score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from localagent.agent.pointer_head import PointerHead
from localagent.data.prompt_contract import render_function_catalog
from localagent.data.render import history_text
from localagent.data.schema import Conversation, Role
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, BPE_EOS, load_tokenizer

SOURCE_URL = "https://huggingface.co/datasets/Team-ACE/ToolACE"
SOURCE_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
SEED = 2033


@dataclass(frozen=True)
class PointerExample:
    prompt: str
    argument: str
    value: str
    start: int
    end: int


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> list[Conversation]:
    rows = [Conversation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty ToolACE input: {path}")
    return rows


def _find_span(ids: list[int], value_ids: list[int]) -> tuple[int, int] | None:
    if not value_ids:
        return None
    for start in range(len(ids) - len(value_ids) + 1):
        if ids[start : start + len(value_ids)] == value_ids:
            return start, start + len(value_ids) - 1
    return None


def _examples(rows: list[Conversation], tokenizer: Any, max_seq_len: int) -> list[PointerExample]:
    examples: list[PointerExample] = []
    for row in rows:
        catalog = render_function_catalog(row.tools) + BPE_EOS
        for index, message in enumerate(row.messages):
            if message.role != Role.assistant or not message.tool_calls:
                continue
            prompt = catalog + history_text(row.messages[:index]) + ASSISTANT
            ids = tokenizer.encode(prompt)
            if len(ids) > max_seq_len:
                ids = ids[-max_seq_len:]
            for call in message.tool_calls:
                for argument, raw_value in call.arguments.items():
                    if not isinstance(raw_value, str) or not raw_value:
                        continue
                    value = str(raw_value)
                    span = _find_span(ids, tokenizer.encode(value))
                    if span is not None:
                        examples.append(
                            PointerExample(
                                prompt=prompt,
                                argument=str(argument),
                                value=value,
                                start=span[0],
                                end=span[1],
                            )
                        )
    if not examples:
        raise ValueError("ToolACE input has no locatable string argument spans")
    return examples


def _pointer_from_state(model: LocalAgentLM, checkpoint: dict[str, Any], args: list[str]) -> PointerHead:
    pointer = PointerHead(model.cfg.d_model, args=args)
    state = checkpoint.get("ptr_head")
    old_args = checkpoint.get("ptr_args") or []
    if not isinstance(state, dict):
        return pointer
    # Some browser-context checkpoints contain a trained legacy pointer matrix but omit the
    # argument-name sidecar. Preserve those rows under deterministic placeholders so the common
    # start/end projections remain loadable; never guess that a legacy row matches a public name.
    arg_weight = state.get("arg_emb.weight")
    if not old_args and isinstance(arg_weight, torch.Tensor) and arg_weight.ndim == 2:
        old_args = [f"__legacy_ptr_{index}" for index in range(int(arg_weight.shape[0]))]
    legacy = PointerHead(model.cfg.d_model, args=old_args)
    legacy.load_state_dict(state)
    pointer.start.load_state_dict(legacy.start.state_dict())
    pointer.end.load_state_dict(legacy.end.state_dict())
    with torch.no_grad():
        for name, index in pointer.arg_idx.items():
            if name in legacy.arg_idx:
                pointer.arg_emb.weight[index].copy_(legacy.arg_emb.weight[legacy.arg_idx[name]])
    return pointer


def _features(model: LocalAgentLM, tokenizer: Any, examples: list[PointerExample]) -> list[torch.Tensor]:
    model.eval()
    cached: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for example in examples:
            if example.prompt not in cached:
                ids = tokenizer.encode(example.prompt)[-model.cfg.max_seq_len :]
                tensor = torch.tensor([ids], dtype=torch.long)
                _, hidden = model(tensor, return_hidden=True)
                cached[example.prompt] = hidden[0].detach()
    return [cached[example.prompt] for example in examples]


def _train(
    model: LocalAgentLM,
    tokenizer: Any,
    examples: list[PointerExample],
    args: list[str],
    *,
    init: PointerHead,
    steps: int,
    batch_size: int,
    lr: float,
) -> PointerHead:
    features = _features(model, tokenizer, examples)
    arg_index = {name: index for index, name in enumerate(args)}
    pointer = init.train()
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=lr)
    rng = random.Random(SEED)
    for _step in range(steps):
        indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
        max_len = max(features[index].shape[0] for index in indices)
        batch = torch.zeros(batch_size, max_len, model.cfg.d_model)
        for row, index in enumerate(indices):
            batch[row, : features[index].shape[0]] = features[index]
        arg_ids = torch.tensor([arg_index[examples[index].argument] for index in indices])
        start, end = pointer.logits(batch, arg_ids)
        for row, index in enumerate(indices):
            start[row, features[index].shape[0] :] = torch.finfo(start.dtype).min
            end[row, features[index].shape[0] :] = torch.finfo(end.dtype).min
        loss = F.cross_entropy(start, torch.tensor([examples[index].start for index in indices]))
        loss = loss + F.cross_entropy(end, torch.tensor([examples[index].end for index in indices]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return pointer.eval()


def _score(model: LocalAgentLM, tokenizer: Any, pointer: PointerHead, examples: list[PointerExample]) -> dict[str, Any]:
    features = _features(model, tokenizer, examples)
    exact = decoded = 0
    covered = 0
    for feature, example in zip(features, examples, strict=True):
        if example.argument not in pointer.arg_idx:
            continue
        covered += 1
        start, end = pointer.predict_span(feature, example.argument)
        exact += int((start, end) == (example.start, example.end))
        ids = tokenizer.encode(example.prompt)[-model.cfg.max_seq_len :]
        predicted = tokenizer.decode(ids[start : end + 1])
        decoded += int(predicted == example.value)
    count = max(1, covered)
    return {
        "examples": len(examples),
        "covered_examples": covered,
        "coverage": covered / max(1, len(examples)),
        "span_exact": exact / count,
        "decoded_value_exact": decoded / count,
        "argument_vocab": len(pointer.args),
    }


def _movement(before: PointerHead, after: PointerHead) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, names in {
        "argument_embedding": ("arg_emb.weight",),
        "start_projection": ("start.weight",),
        "end_projection": ("end.weight",),
    }.items():
        base = torch.cat([before.state_dict()[name].float().flatten() for name in names])
        delta = torch.cat(
            [(after.state_dict()[name].float() - before.state_dict()[name].float()).flatten() for name in names]
        )
        result[group] = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite pointer-probe outputs")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, and learning rate must be positive")

    train_rows = _load(args.train)
    eval_rows = _load(args.eval)
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    model.eval()
    metadata = parent.get("tokenizer", {"kind": "byte"})
    tokenizer = load_tokenizer(metadata["kind"], metadata.get("path"))
    train_examples = _examples(train_rows, tokenizer, config.max_seq_len)
    eval_examples = _examples(eval_rows, tokenizer, config.max_seq_len)
    frequencies: dict[str, int] = {}
    for example in train_examples:
        frequencies[example.argument] = frequencies.get(example.argument, 0) + 1
    argument_vocab = sorted(frequencies)
    inherited = _pointer_from_state(model, parent, argument_vocab).eval()
    inherited_metrics = _score(model, tokenizer, inherited, eval_examples)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    transferred = _train(
        model,
        tokenizer,
        train_examples,
        argument_vocab,
        init=_pointer_from_state(model, parent, argument_vocab),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    transferred_metrics = _score(model, tokenizer, transferred, eval_examples)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    random_model = LocalAgentLM(config)
    random_model.eval()
    random_pointer = _train(
        random_model,
        tokenizer,
        train_examples,
        argument_vocab,
        init=PointerHead(config.d_model, args=argument_vocab),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    random_metrics = _score(random_model, tokenizer, random_pointer, eval_examples)

    child = dict(parent)
    child.update(
        {
            "ptr_head": transferred.state_dict(),
            "ptr_args": list(argument_vocab),
            "stage": "sft_toolace_action_history_pointer_probe",
            "parent_checkpoint_sha256": _identity(args.init)["sha256"],
            "toolace_action_history_pointer_training": {
                "dataset": "Team-ACE/ToolACE",
                "url": SOURCE_URL,
                "revision": SOURCE_REVISION,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_locatable_spans": len(train_examples),
                "eval_locatable_spans": len(eval_examples),
                "argument_vocab": list(argument_vocab),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": SEED,
                "backbone_frozen": True,
                "inherited_eval": inherited_metrics,
                "transferred_eval": transferred_metrics,
                "random_eval": random_metrics,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)

    report = {
        "kind": "localagent_toolace_action_history_pointer_transfer_probe",
        "schema_version": 1,
        "source": {
            "dataset": "Team-ACE/ToolACE",
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_locatable_spans": len(train_examples),
            "eval_locatable_spans": len(eval_examples),
        },
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": SEED,
            "backbone_frozen": True,
            "context": "catalog + BPE_EOS + action-history + ASSISTANT",
        },
        "arms": {
            "inherited_pretrained_pointer": {
                "backbone": "m173_warm_frozen",
                "pointer": "inherited_common_rows_plus_new_arg_rows",
                "metrics": inherited_metrics,
            },
            "retrained_pretrained_backbone": {
                "backbone": "m173_warm_frozen",
                "pointer": "retrained_on_locatable_toolace_spans",
                "metrics": transferred_metrics,
                "pointer_relative_movement": _movement(inherited, transferred),
            },
            "retrained_matched_random_backbone": {
                "backbone": "matched_random_frozen",
                "pointer": "retrained_on_locatable_toolace_spans",
                "metrics": random_metrics,
            },
        },
        "decision": {
            "transfer_improves_over_inherited_decoded_value": transferred_metrics["decoded_value_exact"]
            > inherited_metrics["decoded_value_exact"],
            "transfer_beats_random_decoded_value": transferred_metrics["decoded_value_exact"]
            > random_metrics["decoded_value_exact"],
            "adoption": "adopt_pointer_only_if_free_run_argument_exact_improves",
            "reason": "The WebGPU-shaped free-run receipt is the deployment gate for pointer adoption.",
        },
        "claim_boundary": (
            "ToolACE source-projected pointer diagnostic over values that occur verbatim in the "
            "catalog/history context; no native execution, official ToolACE/BFCL score, or external side effects."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
