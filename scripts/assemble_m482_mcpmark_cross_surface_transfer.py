#!/usr/bin/env python3
"""Seal the larger public MCPMark Notion/browser/filesystem transfer ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assemble(*, warm: Path, random: Path, comparison: Path, train_metadata: Path) -> dict[str, Any]:
    warm_report = _load(warm)
    random_report = _load(random)
    comparison_report = _load(comparison)
    train_manifest = _load(train_metadata)
    if warm_report.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("warm report kind mismatch")
    if random_report.get("kind") != warm_report.get("kind"):
        raise ValueError("random report kind mismatch")
    if comparison_report.get("kind") != "localagent_cross_surface_transfer_ablation_report":
        raise ValueError("comparison kind mismatch")
    if warm_report["parent"]["sha256"] != random_report["parent"]["sha256"]:
        raise ValueError("matched arms use different parents")
    if comparison_report["parent"]["sha256"] != warm_report["parent"]["sha256"]:
        raise ValueError("comparison parent mismatch")
    if train_manifest.get("dataset") != "Jakumetsu/mcpmark-trajectory-log":
        raise ValueError("unexpected productivity dataset")
    if warm_report["rows"] != {"train": 11, "eval": 12}:
        raise ValueError(f"unexpected row counts: {warm_report['rows']}")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_cross_surface_transfer_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "sources": {
            "trajectory_log": {
                "dataset": "Jakumetsu/mcpmark-trajectory-log",
                "url": "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log",
                "revision": train_manifest["revision"],
                "training_manifest": _identity(train_metadata),
            },
            "mcpmark": {
                "dataset": "MCPMark",
                "url": "https://github.com/eval-sys/mcpmark",
                "revision": "cd45b7f57923b9b3985467f5139927575f83141c",
                "official_split_verified": False,
            },
        },
        "parent": warm_report["parent"],
        "training": {
            "warm_report": _identity(warm),
            "random_report": _identity(random),
            "comparison": _identity(comparison),
            "train_sources": warm_report["train_sources"],
            "eval_sources": warm_report["eval_sources"],
            "rows": warm_report["rows"],
            "hyperparameters": warm_report["hyperparameters"],
            "split_contract": warm_report["split_contract"],
        },
        "aggregate": comparison_report["aggregate"],
        "surfaces": comparison_report["surfaces"],
        "transfer_decision": comparison_report["decision"],
        "weight_groups": {
            "warm": comparison_report["warm_weight_groups"],
            "random": comparison_report["random_weight_groups"],
        },
        "claim_boundary": (
            "Public MCPMark redacted Notion/Playwright/filesystem text-and-tool-sequence transfer with "
            "source-local parent-disjoint holdouts and a matched warm/random ablation. This is not an "
            "official MCPMark score, native browser/MCP execution, screenshot-grounding result, or "
            "evidence of email/Notion side effects."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = assemble(
        warm=args.warm,
        random=args.random,
        comparison=args.comparison,
        train_metadata=args.train_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
