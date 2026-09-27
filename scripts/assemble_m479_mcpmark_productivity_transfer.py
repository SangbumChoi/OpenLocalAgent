#!/usr/bin/env python3
"""Seal the MCPMark Notion/Playwright warm-vs-random transfer experiment.

The inputs are local, redacted Conversation/report artifacts produced from the pinned public
MCPMark trajectory log.  The resulting receipt keeps the provenance and native replay boundary
explicit: the one-task Playwright verifier is diagnostic and is not an official MCPMark score.
"""

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


def assemble(
    *,
    warm: Path,
    random: Path,
    comparison: Path,
    train_metadata: Path,
    native: Path,
) -> dict[str, Any]:
    warm_report = _load(warm)
    random_report = _load(random)
    comparison_report = _load(comparison)
    train_manifest = _load(train_metadata)
    native_report = _load(native)
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
        raise ValueError("unexpected trajectory dataset")
    if native_report.get("benchmark_id") != "mcpmark":
        raise ValueError("native receipt benchmark mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_productivity_transfer_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "source": {
            "dataset": "Jakumetsu/mcpmark-trajectory-log",
            "url": "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log",
            "revision": train_manifest["revision"],
            "license_note": "Use the public dataset's published license and audit retained arguments before redistribution.",
            "training_manifest": _identity(train_metadata),
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
        },
        "transfer_decision": comparison_report["decision"],
        "aggregate": comparison_report["aggregate"],
        "surfaces": comparison_report["surfaces"],
        "weight_groups": {
            "warm": comparison_report["warm_weight_groups"],
            "random": comparison_report["random_weight_groups"],
        },
        "native_replay": {
            "receipt": _identity(native),
            "summary": native_report.get("summary"),
            "environment": native_report.get("environment"),
            "status": "blocked_by_missing_network_dependency",
            "reason": "@playwright/mcp@0.0.68 could not be fetched in the sandbox (npm ENOTFOUND).",
        },
        "claim_boundary": (
            "Public MCPMark redacted Notion/Playwright text-and-tool-sequence transfer with a matched "
            "warm/random ablation. This is not an official MCPMark split or score, does not include "
            "tool outputs, screenshot grounding, live email/Notion side effects, or real accounts, "
            "and the native replay was blocked by an unavailable npm dependency."
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
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    receipt = assemble(
        warm=args.warm,
        random=args.random,
        comparison=args.comparison,
        train_metadata=args.train_metadata,
        native=args.native,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
