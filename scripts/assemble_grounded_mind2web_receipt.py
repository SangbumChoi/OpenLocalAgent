#!/usr/bin/env python3
"""Assemble a hash-bound matched warm/random grounded-Mind2Web transfer receipt.

The trainer writes one report per checkpoint arm.  This audit joins those reports with the exact
normalized train/eval files, verifies parent-record and typed-slot disjointness, and records only
metadata plus metrics.  It never copies prompts or DOM snapshots into the tracked receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Role


DATASET = "osunlp/Mind2Web"
SOURCE_URL = "https://huggingface.co/datasets/osunlp/Mind2Web"
SOURCE_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _rows(path: Path) -> list[Conversation]:
    rows = [
        Conversation.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no conversations in {path}")
    return rows


def _audit_split(train: list[Conversation], evaluation: list[Conversation]) -> dict[str, Any]:
    train_ids = {str(row.meta.get("parent_record_id")) for row in train}
    eval_ids = {str(row.meta.get("parent_record_id")) for row in evaluation}
    id_overlap = sorted((train_ids & eval_ids) - {"None"})
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
    slot_overlap = sorted(train_slots & eval_slots)
    if id_overlap or slot_overlap:
        raise ValueError(f"Mind2Web split leakage: ids={id_overlap[:5]}, slots={slot_overlap[:5]}")

    def decisions(rows: list[Conversation]) -> int:
        return sum(
            len(message.tool_calls)
            for row in rows
            for message in row.messages
            if message.role == Role.assistant
        )

    return {
        "train_conversations": len(train),
        "eval_conversations": len(evaluation),
        "train_parent_records": len(train_ids - {"None"}),
        "eval_parent_records": len(eval_ids - {"None"}),
        "train_decisions": decisions(train),
        "eval_decisions": decisions(evaluation),
        "train_parent_record_ids_sha256": hashlib.sha256(
            "\n".join(sorted(train_ids)).encode("utf-8")
        ).hexdigest(),
        "eval_parent_record_ids_sha256": hashlib.sha256(
            "\n".join(sorted(eval_ids)).encode("utf-8")
        ).hexdigest(),
        "task_id_disjoint": True,
        "typed_slot_disjoint": True,
    }


def _weight_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    groups = report.get("groups", {})
    return {
        "path": str(path),
        "sha256": _identity(path)["sha256"],
        "compatibility": report.get("compatibility", {}),
        "relative_delta_l2": {
            key: groups.get(key, {}).get("relative_delta_l2")
            for key in ("embedding", "attention_or_mixer", "ffn", "normalization", "action_heads")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--warm-weight", type=Path, required=True)
    parser.add_argument("--random-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")

    train, evaluation = _rows(args.train), _rows(args.eval)
    split = _audit_split(train, evaluation)
    warm = json.loads(args.warm_report.read_text(encoding="utf-8"))
    random_arm = json.loads(args.random_report.read_text(encoding="utf-8"))
    warm_exact = float(warm["after"]["exact_span"])
    random_exact = float(random_arm["after"]["exact_span"])
    receipt: dict[str, Any] = {
        "kind": "localagent_grounded_mind2web_transfer_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "license": "CC-BY-4.0",
        "source": _identity(args.source),
        "manifest": _identity(args.manifest) if args.manifest else None,
        "inputs": {"train": _identity(args.train), "eval": _identity(args.eval)},
        "split_audit": split,
        "training": {
            "warm": {
                "report": _identity(args.warm_report),
                "parent": warm.get("parent"),
                "child": warm.get("child"),
                "rows": warm.get("rows"),
                "hyperparameters": warm.get("hyperparameters"),
                "before": warm.get("before"),
                "after": warm.get("after"),
                "weight_transfer": _weight_summary(args.warm_weight),
            },
            "random": {
                "report": _identity(args.random_report),
                "parent": random_arm.get("parent"),
                "child": random_arm.get("child"),
                "rows": random_arm.get("rows"),
                "hyperparameters": random_arm.get("hyperparameters"),
                "before": random_arm.get("before"),
                "after": random_arm.get("after"),
                "weight_transfer": _weight_summary(args.random_weight),
            },
        },
        "comparison": {
            "warm_minus_random_exact_span": warm_exact - random_exact,
            "warm_exact_span": warm_exact,
            "random_exact_span": random_exact,
        },
        "compatibility_policy": (
            "The two additional pointer-argument rows are an intentional BROWSER_PTR_ARGS vocabulary "
            "expansion; tokenizer/config compatibility is required, while the ptr_head.arg_emb shape "
            "difference is expected and is not treated as a backbone mismatch."
        ),
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Public Mind2Web train-split DOM-enriched continuation on a task- and typed-slot-disjoint "
            "in-source holdout. This is pointer-span/action replay evidence only: it is not the "
            "official Mind2Web test score, BrowserGym/WebArena success, screenshot-grounded native "
            "control, real browser account, email, Notion, MCP, or external side effect."
        ),
    }
    receipt["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
