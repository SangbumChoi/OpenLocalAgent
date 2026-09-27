#!/usr/bin/env python3
"""Run a bounded ToolSandbox projection transfer and tool-name masking probe.

This is a public-data diagnostic, not an official ToolSandbox benchmark run.  It uses the
canonical public-source projection already produced by the ToolSandbox adapter, filters rows that
do not satisfy their declared JSON schema (the projection intentionally retains those rows for
coverage accounting), and trains a short LM-only SFT arm with the full-catalog prompt contract.
The full catalog is important here: opaque aliases are copyable from the prompt, whereas a legacy
prompt would make a per-row alias unknowable.  Warm-start and matched-random arms are evaluated on
the unmasked and deterministically masked public eval rows, with tensor-group movement recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1, schema_matches, validate_tool_catalog
from localagent.data.schema import Conversation, Role
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.function_masking import augment_conversations
from localagent.train.sft import _evaluate_conversations, sft

SOURCE_URL = "https://github.com/apple/ToolSandbox"
DEFAULT_SEED = 2058
MASKING_CONFIG = {"enabled": True, "mask_fraction": 1.0, "variants": 1, "name_prefix": "opaque"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _load(path: Path) -> list[Conversation]:
    rows = [
        Conversation.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty conversation source: {path}")
    return rows


def _schema_valid(row: Conversation) -> bool:
    registry = validate_tool_catalog(row.tools)
    for message in row.messages:
        if message.role != Role.assistant:
            continue
        for call in message.tool_calls:
            tool = registry.get(call.name)
            if tool is None or not schema_matches(call.arguments, tool.parameters):
                return False
    return True


def _valid_rows(rows: Sequence[Conversation]) -> tuple[list[Conversation], int]:
    valid = [row for row in rows if _schema_valid(row)]
    return valid, len(rows) - len(valid)


def _relative_l2(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in before.items():
        if name not in after:
            continue
        left = value.detach().float()
        right = after[name].detach().float()
        numerator += float((right - left).pow(2).sum())
        denominator += float(left.pow(2).sum())
    return (numerator**0.5) / max(denominator**0.5, 1e-12)


def _movement_groups(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    groups: dict[str, dict[str, torch.Tensor]] = {
        "embedding": {},
        "mixer": {},
        "ffn": {},
        "normalization": {},
    }
    for name, value in before.items():
        if name.startswith("embed."):
            groups["embedding"][name] = value
        elif ".attn." in name:
            groups["mixer"][name] = value
        elif ".ffn." in name:
            groups["ffn"][name] = value
        elif "norm." in name or name == "norm.weight":
            groups["normalization"][name] = value
    return {
        "backbone": _relative_l2(before, after),
        **{
            group: _relative_l2(values, {name: after[name] for name in values})
            for group, values in groups.items()
        },
    }


def _evaluate(
    model: LocalAgentLM,
    tokenizer: Any,
    rows: Sequence[Conversation],
    *,
    limit: int,
) -> dict[str, Any]:
    subset = list(rows[:limit])
    return _evaluate_conversations(
        model,
        subset,
        tokenizer,
        max_seq_len=model.cfg.max_seq_len,
        batch_size=4,
        device="cpu",
        amp_dtype=torch.float32,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )


def _arm(
    *,
    initial_state: Mapping[str, torch.Tensor],
    cfg: ModelConfig,
    tokenizer: Any,
    train_rows: Sequence[Conversation],
    eval_rows: Sequence[Conversation],
    eval_masked_rows: Sequence[Conversation],
    train_eval_rows: Sequence[Conversation],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    warm_start: bool,
    eval_limit: int,
) -> dict[str, Any]:
    torch.manual_seed(seed + (0 if warm_start else 1))
    model = LocalAgentLM(cfg)
    if warm_start:
        model.load_state_dict(initial_state)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before = {
        "eval_canonical": _evaluate(model, tokenizer, eval_rows, limit=eval_limit),
        "eval_masked": _evaluate(model, tokenizer, eval_masked_rows, limit=eval_limit),
        "train_masked": _evaluate(model, tokenizer, train_eval_rows, limit=eval_limit),
    }
    history, _, _, metrics = sft(
        model,
        [],
        tokenizer,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        warmup=max(1, min(steps // 4, 4)),
        weight_decay=0.0,
        grad_clip=1.0,
        device="cpu",
        joint_tool_head=False,
        conversations=list(train_rows),
        conversation_sources=["toolsandbox_projection"] * len(train_rows),
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        shuffle=True,
        seed=seed,
        amp_dtype=torch.float32,
        return_metrics=True,
    )
    after = {
        "eval_canonical": _evaluate(model, tokenizer, eval_rows, limit=eval_limit),
        "eval_masked": _evaluate(model, tokenizer, eval_masked_rows, limit=eval_limit),
        "train_masked": _evaluate(model, tokenizer, train_eval_rows, limit=eval_limit),
    }
    return {
        "kind": "warm_start" if warm_start else "matched_random",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "loss_first": float(history[0]),
        "loss_last": float(history[-1]),
        "before": before,
        "after": after,
        "weight_movement_relative_l2": _movement_groups(initial, model.state_dict()),
        "sft_metrics": metrics,
    }


def _self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    if args.steps < 1 or args.batch_size < 1 or args.eval_limit < 1 or args.lr <= 0:
        raise SystemExit("steps, batch-size, eval-limit, and lr must be positive")
    for path in (args.init, args.train, args.eval, args.source_manifest):
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["cfg"])
    cfg.assert_within_budget()
    tokenizer = load_tokenizer("bpe", path="data/tokenizer-webgpu-proxy-16k.json")
    initial_model = LocalAgentLM(cfg)
    initial_model.load_state_dict(checkpoint["state_dict"])
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}

    raw_train = _load(args.train)
    raw_eval = _load(args.eval)
    train_rows, train_invalid = _valid_rows(raw_train)
    eval_rows, eval_invalid = _valid_rows(raw_eval)
    if not train_rows or not eval_rows:
        raise SystemExit("schema-valid ToolSandbox projection rows are required for the probe")
    masked_train, train_indices, train_audit = augment_conversations(
        train_rows, MASKING_CONFIG, seed=args.seed
    )
    masked_eval, _, eval_audit = augment_conversations(
        eval_rows, MASKING_CONFIG, seed=args.seed
    )
    if not masked_train or not masked_eval:
        raise SystemExit("function masking produced no rows")

    # Keep the diagnostic evaluation bounded while retaining deterministic source order.
    train_eval_rows = masked_train[: args.eval_limit]
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest_sha = source_manifest.get("manifest_self_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise SystemExit("source manifest is missing its self hash")

    warm = _arm(
        initial_state=initial_state,
        cfg=cfg,
        tokenizer=tokenizer,
        train_rows=masked_train,
        eval_rows=eval_rows,
        eval_masked_rows=masked_eval,
        train_eval_rows=train_eval_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        warm_start=True,
        eval_limit=args.eval_limit,
    )
    random_arm = _arm(
        initial_state=initial_state,
        cfg=cfg,
        tokenizer=tokenizer,
        train_rows=masked_train,
        eval_rows=eval_rows,
        eval_masked_rows=masked_eval,
        train_eval_rows=train_eval_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        warm_start=False,
        eval_limit=args.eval_limit,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "toolsandbox_function_masking_transfer_pilot",
        "source": {
            "dataset": "apple/ToolSandbox",
            "source_url": SOURCE_URL,
            "manifest": _identity(args.source_manifest),
            "manifest_self_sha256": manifest_sha,
            "train": _identity(args.train),
            "eval": _identity(args.eval),
            "revision": source_manifest.get("revision"),
            "projection_kind": source_manifest.get("kind"),
        },
        "checkpoint": {
            "path": str(args.init),
            "sha256": _sha256_file(args.init),
            "params": cfg.estimate_params(),
            "config": cfg.__dict__,
            "tokenizer": _identity(Path("data/tokenizer-webgpu-proxy-16k.json")),
        },
        "data": {
            "raw_train_rows": len(raw_train),
            "raw_eval_rows": len(raw_eval),
            "schema_valid_train_rows": len(train_rows),
            "schema_invalid_train_rows": train_invalid,
            "schema_valid_eval_rows": len(eval_rows),
            "schema_invalid_eval_rows": eval_invalid,
            "masked_train_rows": len(masked_train),
            "masked_eval_rows": len(masked_eval),
            "masking": {"train": train_audit, "eval": eval_audit},
            "prompt_contract": OPENAI_FULL_CATALOG_V1,
        },
        "protocol": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eval_limit": args.eval_limit,
            "seed": args.seed,
            "device": "cpu",
            "arms": ["warm_start", "matched_random"],
        },
        "arms": {"warm_start": warm, "matched_random": random_arm},
        "claim_boundary": (
            "Diagnostic only: this is a schema-valid subset of the public ToolSandbox scenario "
            "projection, not the official ToolSandbox split or leaderboard. No ToolSandbox "
            "simulator, verifier, user simulator, external API, mobile runtime, or desktop "
            "environment was executed. Invalid projection rows are reported and excluded from "
            "the full-catalog renderer because their declared required arguments are incomplete."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "receipt_self_sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
