#!/usr/bin/env python3
"""Train a metadata-only CUA-Gym surface probe with a matched random control.

CUA-Gym's public task table contains instructions and platform metadata, but it does not publish
an official held-out action split.  This script therefore trains only a small linear probe for the
metadata field ``platform`` (desktop/web/cross_app).  It deliberately does not read setup files,
reward code, screenshots, or action traces.  The result measures whether a frozen checkpoint's
prompt representation separates broad deployment surfaces; it is not task success, action
supervision, or native desktop/browser evidence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.agent.tool_head import _feat
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
try:  # Direct ``python scripts/...`` invocation keeps ``scripts`` off sys.path.
    from scripts.profile_cua_gym_metadata import _load_rows
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI entry point
    from profile_cua_gym_metadata import _load_rows


LABELS = ["desktop", "web", "cross_app"]
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
DATASET = "xlangai/CUA-Gym"
SOURCE_URL = "https://huggingface.co/datasets/xlangai/CUA-Gym"
SOURCE_REVISION = "3c021d0"


class SurfaceHead(nn.Module):
    """Small probe kept separate from the model's fixed five-way route head."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, len(LABELS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _surface(row: dict[str, Any]) -> str | None:
    platform = row.get("platform")
    return str(platform) if platform in LABEL_INDEX else None


def _select_rows(rows: list[dict[str, Any]], max_per_label: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for row in rows:
        label = _surface(row)
        if label is not None:
            grouped[label].append(row)
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        candidates = sorted(grouped[label], key=lambda row: str(row["id"]))
        selected.extend(candidates[:max_per_label])
    return sorted(selected, key=lambda row: str(row["id"]))


def _split(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by task identity, not row order or prompt text."""

    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{row['id']}".encode("utf-8")).hexdigest()
        (evaluation if int(digest[:8], 16) % 5 == 0 else train).append(row)
    if not train or not evaluation:
        raise ValueError("CUA-Gym split unexpectedly produced an empty train/eval partition")
    if {str(row["id"]) for row in train} & {str(row["id"]) for row in evaluation}:
        raise ValueError("CUA-Gym task-id split overlap")
    return train, evaluation


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(collections.Counter(_surface(row) for row in rows).items()))


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], LocalAgentLM, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer_info = checkpoint.get("tokenizer") or {"kind": "byte"}
    tokenizer = load_tokenizer(tokenizer_info["kind"], tokenizer_info.get("path"))
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint tokenizer and model vocabulary differ")
    model.eval()
    return checkpoint, model, tokenizer


@torch.no_grad()
def _features(model: LocalAgentLM, tokenizer: Any, rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([_feat(model, tokenizer, str(row["instruction"]), "cpu") for row in rows])


def _metrics(head: SurfaceHead, features: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = torch.tensor([LABEL_INDEX[_surface(row)] for row in rows])
    with torch.no_grad():
        predictions = head(features).argmax(-1)
    per_label: dict[str, dict[str, int | float]] = {}
    for label, index in LABEL_INDEX.items():
        mask = labels == index
        total = int(mask.sum())
        correct = int(((predictions == labels) & mask).sum())
        per_label[label] = {
            "rows": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
    return {
        "rows": len(rows),
        "accuracy": float((predictions == labels).float().mean().item()),
        "balanced_accuracy": sum(float(item["accuracy"]) for item in per_label.values()) / len(LABELS),
        "per_label": per_label,
    }


def _train_head(
    features: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    d_model: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[SurfaceHead, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    head = SurfaceHead(d_model)
    initial = {name: value.detach().clone() for name, value in head.state_dict().items()}
    labels = torch.tensor([LABEL_INDEX[_surface(row)] for row in rows])
    counts = torch.bincount(labels, minlength=len(LABELS)).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).div(len(LABELS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    generator = random.Random(seed)
    history: list[float] = []
    for _step in range(steps):
        indices = torch.tensor([generator.randrange(len(rows)) for _ in range(batch_size)])
        logits = head(features[indices])
        loss = F.cross_entropy(logits, labels[indices], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    movement: dict[str, float] = {}
    for name, value in head.state_dict().items():
        denominator = float(initial[name].norm().item())
        movement[name] = float((value - initial[name]).norm().item()) / max(denominator, 1.0e-12)
    return head, {"initial_loss": history[0], "final_loss": history[-1]}, movement


def _run_arm(
    checkpoint_path: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    checkpoint, model, tokenizer = _load_checkpoint(checkpoint_path)
    train_features = _features(model, tokenizer, train_rows)
    eval_features = _features(model, tokenizer, eval_rows)
    head, loss, movement = _train_head(
        train_features,
        train_rows,
        d_model=model.cfg.d_model,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    torch.save(
        {
            "kind": "localagent_cua_gym_surface_probe_head",
            "schema_version": 1,
            "parent_checkpoint_sha256": _identity(checkpoint_path)["sha256"],
            "labels": LABELS,
            "head": head.state_dict(),
        },
        output,
    )
    return {
        "checkpoint": _identity(checkpoint_path),
        "probe_head": _identity(output),
        "checkpoint_stage": checkpoint.get("stage"),
        "train": _metrics(head, train_features, train_rows),
        "eval": _metrics(head, eval_features, eval_rows),
        "optimization": loss,
        "probe_head_relative_l2": movement,
        "backbone_updated": False,
        "backbone_relative_l2": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--random-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-output", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--head-seed", type=int, default=2029)
    parser.add_argument("--max-per-label", type=int, default=1500)
    parser.add_argument("--head-steps", type=int, default=300)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=5.0e-3)
    args = parser.parse_args()
    outputs = [args.warm_output, args.random_output, args.report]
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite probe outputs")
    if args.max_per_label < 1 or args.head_steps < 1 or args.head_batch_size < 1 or args.head_lr <= 0:
        raise SystemExit("probe limits and optimizer settings must be positive")

    rows = _load_rows(args.input)
    selected = _select_rows(rows, args.max_per_label)
    train_rows, eval_rows = _split(selected, args.split_seed)
    if set(_counts(train_rows)) != set(LABELS) or set(_counts(eval_rows)) != set(LABELS):
        raise ValueError("every surface label must appear in both partitions")
    warm = _run_arm(
        args.warm_checkpoint,
        train_rows,
        eval_rows,
        steps=args.head_steps,
        batch_size=args.head_batch_size,
        lr=args.head_lr,
        seed=args.head_seed,
        output=args.warm_output,
    )
    random_arm = _run_arm(
        args.random_checkpoint,
        train_rows,
        eval_rows,
        steps=args.head_steps,
        batch_size=args.head_batch_size,
        lr=args.head_lr,
        seed=args.head_seed,
        output=args.random_output,
    )
    report = {
        "kind": "localagent_cua_gym_surface_probe",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "license": "CC-BY-4.0",
        "source": _identity(args.input),
        "selection": {
            "all_metadata_rows": len(rows),
            "selected_rows": len(selected),
            "max_per_label": args.max_per_label,
            "labels": LABELS,
            "selected_counts": _counts(selected),
            "train_counts": _counts(train_rows),
            "eval_counts": _counts(eval_rows),
            "split_seed": args.split_seed,
            "task_id_disjoint": True,
        },
        "hyperparameters": {
            "head_seed": args.head_seed,
            "head_steps": args.head_steps,
            "head_batch_size": args.head_batch_size,
            "head_lr": args.head_lr,
            "backbone_frozen": True,
        },
        "arms": {"warm": warm, "random": random_arm},
        "comparison": {
            "eval_accuracy_delta_warm_minus_random": warm["eval"]["accuracy"] - random_arm["eval"]["accuracy"],
            "eval_balanced_accuracy_delta_warm_minus_random": warm["eval"]["balanced_accuracy"]
            - random_arm["eval"]["balanced_accuracy"],
        },
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Metadata-derived platform classification over a deterministic task-id holdout. The probe "
            "uses CUA-Gym instructions but no setup/reward artifacts, screenshots, or action traces. "
            "It is not CUA-Gym task success, RLVR, native desktop/browser control, email/Notion/MCP "
            "side effects, or evidence that warm weights improve downstream agent success."
        ),
    }
    report["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
