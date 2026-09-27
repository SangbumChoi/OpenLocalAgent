#!/usr/bin/env python3
"""Assemble a compact, hash-bound receipt for a matched current-child continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation
from scripts.compare_cross_surface_controls import compare


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return value


def _rows(path: Path) -> list[Conversation]:
    with path.open(encoding="utf-8") as handle:
        return [Conversation.from_json(line) for line in handle if line.strip()]


def _split_audit(warm: dict[str, Any]) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    train_parent_sets: dict[str, set[str]] = {}
    eval_parent_sets: dict[str, set[str]] = {}
    for source in warm["train_sources"]:
        label = source["label"]
        rows = _rows(Path(source["input"]["path"]))
        train_parent_sets[label] = {
            str(row.meta["parent_record_id"])
            for row in rows
            if row.meta.get("parent_record_id")
        }
        per_source.setdefault(label, {})["train_rows"] = len(rows)
        per_source[label]["train_parent_records"] = len(train_parent_sets[label])
    for source in warm["eval_sources"]:
        label = source["label"]
        rows = _rows(Path(source["input"]["path"]))
        eval_parent_sets[label] = {
            str(row.meta["parent_record_id"])
            for row in rows
            if row.meta.get("parent_record_id")
        }
        per_source.setdefault(label, {})["eval_rows"] = len(rows)
        per_source[label]["eval_parent_records"] = len(eval_parent_sets[label])

    overlaps = {
        label: sorted(train_parent_sets.get(label, set()) & eval_parent_sets.get(label, set()))
        for label in set(train_parent_sets) | set(eval_parent_sets)
        if train_parent_sets.get(label, set()) & eval_parent_sets.get(label, set())
    }
    cross_source_collisions = []
    labels = sorted(train_parent_sets)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            shared = train_parent_sets[left] & train_parent_sets[right]
            if shared:
                cross_source_collisions.append(
                    {"left": left, "right": right, "parent_records": sorted(shared)}
                )
    return {
        "per_source": per_source,
        "train_eval_parent_record_overlap": overlaps,
        "cross_source_train_parent_collisions": cross_source_collisions,
        "parent_record_disjoint": not overlaps and not cross_source_collisions,
        "visual_input_omitted_rows": {
            source["label"]: source["visual_input_omitted_rows"]
            for source in warm["train_sources"]
            + warm["eval_sources"]
        },
    }


def _arm(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "report": _identity(report_path),
        "child": report["child"],
        "backbone_init": report["hyperparameters"]["backbone_init"],
        "before": report["before"],
        "after": report["after"],
        "weight_transfer": report["weight_transfer"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    warm = _load_report(args.warm_report)
    random = _load_report(args.random_report)
    comparison = compare(warm, random)
    payload: dict[str, Any] = {
        "kind": "localagent_current_child_cross_surface_transfer_receipt",
        "schema_version": 1,
        "measurement": "m168_current_child_mobile_desktop_browser_mcp_transfer",
        "generated_at": "2026-08-03",
        "rows": warm["rows"],
        "train_sources": warm["train_sources"],
        "eval_sources": warm["eval_sources"],
        "split_audit": _split_audit(warm),
        "parent": warm["parent"],
        "training": {
            "hyperparameters": warm["hyperparameters"],
            "warm_parent_backbone": _arm(warm, args.warm_report),
            "random_backbone_control": _arm(random, args.random_report),
        },
        "comparison": {
            "aggregate": comparison["aggregate"],
            "surfaces": comparison["surfaces"],
            "decision": comparison["decision"],
        },
        "compatibility": {
            "warm": warm["weight_transfer"]["compatibility"],
            "random": random["weight_transfer"]["compatibility"],
        },
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Source-record-disjoint public-train-only continuation over AndroidControl text and "
            "accessibility rows, AgentNet desktop actions, Mind2Web browser grounding rows, and "
            "redacted MCPMark trajectories. This is not an official benchmark score, native "
            "Android/desktop/browser/MCP execution, screenshot grounding, or a real email/Notion "
            "side effect."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["receipt_self_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
