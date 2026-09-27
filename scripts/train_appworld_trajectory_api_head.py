#!/usr/bin/env python3
"""Train a frozen-backbone AppWorld API head from compact multi-turn trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.tool_head import _feat
from localagent.data.schema import Conversation
from localagent.eval.appworld_api_head import AppWorldAPIHead, save_appworld_api_head
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, load_tokenizer
from localagent.train.stage_data import history_text

_CODE_RE = re.compile(r"^apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\(")


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _rows(path: Path) -> list[Conversation]:
    return [Conversation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _examples(rows: list[Conversation]) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    labels: list[str] = []
    for row in rows:
        for index, message in enumerate(row.messages):
            if message.role.value != "assistant" or not message.tool_calls:
                continue
            call = message.tool_calls[0]
            code = str(call.arguments.get("code", ""))
            match = _CODE_RE.match(code.strip())
            if match is None:
                continue
            prompts.append(history_text(row.messages[:index]) + ASSISTANT)
            labels.append(f"{match.group(1)}.{match.group(2)}")
    if not prompts:
        raise ValueError("no trajectory API examples found")
    return prompts, labels


def _features(model: LocalAgentLM, tokenizer: Any, prompts: list[str]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.stack([_feat(model, tokenizer, prompt, "cpu", framed=True) for prompt in prompts])


def _metrics(head: AppWorldAPIHead, features: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    index = {name: i for i, name in enumerate(head.classes)}
    seen = [(feature, label) for feature, label in zip(features, labels) if label in index]
    if not seen:
        return {"rows": 0, "exact": 0, "accuracy": 0.0, "unseen_labels": sorted(set(labels))}
    values = torch.stack([item[0] for item in seen])
    target = torch.tensor([index[item[1]] for item in seen])
    with torch.no_grad():
        predictions = head(values).argmax(-1)
    return {
        "rows": len(seen),
        "exact": int((predictions == target).sum().item()),
        "accuracy": float((predictions == target).float().mean().item()),
        "unseen_labels": sorted(set(labels) - set(index)),
    }


def train(
    *, data: Path, eval_data: Path, init: Path, output: Path, report: Path,
    steps: int, batch_size: int, lr: float,
) -> dict[str, Any]:
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("steps and batch-size must be positive; lr must be positive")
    train_rows, eval_rows = _rows(data), _rows(eval_data)
    parent = torch.load(init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(parent["state_dict"])
    metadata = parent.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(str(metadata.get("kind", "byte")), metadata.get("path"))
    train_prompts, train_labels = _examples(train_rows)
    eval_prompts, eval_labels = _examples(eval_rows)
    classes = tuple(sorted(set(train_labels)))
    train_features = _features(model, tokenizer, train_prompts)
    eval_features = _features(model, tokenizer, eval_prompts)
    head = AppWorldAPIHead(model.cfg.d_model, classes)
    labels = torch.tensor([classes.index(label) for label in train_labels])
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(2027)
    for step in range(steps):
        indices = torch.randint(len(train_labels), (batch_size,), generator=generator)
        loss = F.cross_entropy(head(train_features[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    parent_id = _identity(init)
    source = {
        "dataset": "appworld_ground_truth_api_trajectory_compact",
        "train_input": _identity(data),
        "eval_input": _identity(eval_data),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
    if report.exists() or output.exists():
        raise FileExistsError("refusing to overwrite API-head output or report")
    save_appworld_api_head(output, head, parent_checkpoint=parent_id, source=source)
    result = {
        "kind": "localagent_appworld_trajectory_api_head_training_report",
        "schema_version": 1,
        "parent": parent_id,
        "child": _identity(output),
        "source": source,
        "classes": list(classes),
        "trajectory_examples": {"train": len(train_labels), "eval": len(eval_labels)},
        "hyperparameters": {"steps": steps, "batch_size": batch_size, "learning_rate": lr, "seed": 2027},
        "metrics": {"train": _metrics(head, train_features, train_labels), "eval": _metrics(head, eval_features, eval_labels)},
        "claim_boundary": (
            "Frozen-backbone API head trained on compact public AppWorld trajectory prefixes. It is a "
            "schema-candidate restriction diagnostic, not a complete policy or official benchmark score."
        ),
    }
    result["report_self_sha256"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
