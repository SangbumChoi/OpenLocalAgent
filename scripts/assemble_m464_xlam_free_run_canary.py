#!/usr/bin/env python3
"""Assemble a current-parent/warm-child xLAM constrained-decoder canary receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET = "product-science/xlam-function-calling-60k-raw"
DATASET_URL = "https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw"
SOURCE_SHA256 = "43db9250b50f44d96d2be31983690e101bd083eefea2a4a327e13a3ed8caeee1"
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    warm = json.loads(args.warm_report.read_text(encoding="utf-8"))
    for label, report in (("parent", parent), ("warm", warm)):
        if report["kind"] != "localagent_xlam_derived_function_calling_eval":
            raise ValueError(f"unexpected {label} report kind")
        if report["source"]["derived_dataset"] != DATASET:
            raise ValueError(f"{label} dataset mismatch")
        if report["source"]["source_file"]["sha256"] != SOURCE_SHA256:
            raise ValueError(f"{label} source shard mismatch")
        if report["rows"] != 8:
            raise ValueError(f"{label} row count mismatch")
        if set(report["modes"]) != {"row_retriever"}:
            raise ValueError(f"{label} must contain only row_retriever metrics")
        if report["candidate_modes"] != ["row_retriever"]:
            raise ValueError(f"{label} evaluator mode mismatch")
    if parent["checkpoint"]["sha256"] != PARENT_SHA256:
        raise ValueError("parent report checkpoint mismatch")
    parent_metrics = parent["modes"]["row_retriever"]
    warm_metrics = warm["modes"]["row_retriever"]
    payload = {
        "kind": "localagent_xlam_current_free_run_row_canary",
        "schema_version": 1,
        "source": {
            "derived_dataset": DATASET,
            "derived_url": DATASET_URL,
            "source_file": parent["source"]["source_file"],
            "rows": 8,
            "official_original_split_verified": False,
            "original_dataset": "Salesforce/xlam-function-calling-60k",
        },
        "protocol": {
            "device": "cpu",
            "candidate_mode": "row_retriever",
            "candidate_policy": (
                "Each row's gold tool list is supplied to the constrained decoder; this is a "
                "first-call candidate upper bound, not the global/runtime catalog path."
            ),
            "first_call_only": True,
            "rows": 8,
        },
        "parent": {
            "checkpoint": parent["checkpoint"],
            "metrics": parent_metrics,
        },
        "warm": {
            "checkpoint": warm["checkpoint"],
            "metrics": warm_metrics,
        },
        "comparison": {
            "parent_first_tool_exact_rate": parent_metrics["first_tool_exact_rate"],
            "warm_first_tool_exact_rate": warm_metrics["first_tool_exact_rate"],
            "parent_first_arguments_exact_rate": parent_metrics["first_arguments_exact_rate"],
            "warm_first_arguments_exact_rate": warm_metrics["first_arguments_exact_rate"],
            "parent_schema_valid_rate": parent_metrics["schema_valid_rate"],
            "warm_schema_valid_rate": warm_metrics["schema_valid_rate"],
            "warm_minus_parent_tool_exact_rate": (
                warm_metrics["first_tool_exact_rate"]
                - parent_metrics["first_tool_exact_rate"]
            ),
            "warm_minus_parent_arguments_exact_rate": (
                warm_metrics["first_arguments_exact_rate"]
                - parent_metrics["first_arguments_exact_rate"]
            ),
        },
        "decision": {
            "adoption": "retain_as_low_rate_candidate_but_not_free_run_promotion",
            "native_replay_required": True,
            "webgpu_export_allowed": False,
            "reason": (
                "The warm child matches the current parent on this eight-row row-retriever "
                "canary: 50% first-tool exact, 0% first-argument exact, and 100% schema-valid. "
                "The canary is an upper-bound candidate policy and is not sufficient to promote "
                "the child to WebGPU or claim runtime/global-catalog competence."
            ),
        },
        "source_artifacts": {
            "parent_report": identity(args.parent_report),
            "warm_report": identity(args.warm_report),
            "evaluator": identity(args.evaluator),
        },
        "claim_boundary": (
            "Bounded first-call constrained-decoder canary on a public xLAM-derived shard. It is "
            "not an official Salesforce xLAM/BFCL score, multi-call score, global retriever score, "
            "native MCP result, live API execution, or browser/email/Notion side-effect claim."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
