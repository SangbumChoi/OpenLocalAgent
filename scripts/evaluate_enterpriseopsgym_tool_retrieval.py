#!/usr/bin/env python
"""Run the leakage-safe EnterpriseOps-Gym name-only tool-retrieval diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.enterpriseopsgym import (
    ENTERPRISEOPSGYM_ADAPTER,
    ENTERPRISEOPSGYM_DATASET,
    ENTERPRISEOPSGYM_REVISION,
    checkpoint_model,
    load_tasks,
    score_tasks,
    summarize_scores,
)


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--distractors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tasks = load_tasks(args.oracle, args.distractors)
    model, tokenizer, selector, checkpoint = checkpoint_model(args.checkpoint, device=args.device)
    scores = score_tasks(model, tokenizer, selector, tasks, device=args.device)
    source_files = {}
    for label, path in (("oracle", args.oracle), ("distractors", args.distractors)):
        size, digest = _sha256(path)
        source_files[label] = {"bytes": size, "sha256": digest}
    checkpoint_size, checkpoint_sha = _sha256(args.checkpoint)
    payload: dict[str, Any] = {
        "kind": "localagent_enterpriseopsgym_tool_retrieval_receipt",
        "schema_version": 1,
        "adapter": ENTERPRISEOPSGYM_ADAPTER,
        "dataset": ENTERPRISEOPSGYM_DATASET,
        "dataset_revision": ENTERPRISEOPSGYM_REVISION,
        "source_files": source_files,
        "checkpoint": {
            "bytes": checkpoint_size,
            "sha256": checkpoint_sha,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "tokenizer_sha256": checkpoint.get("tokenizer", {}).get("sha256"),
        },
        "protocol": {
            "candidate_mode": "plus_15_tools",
            "oracle_mode": "oracle",
            "records": len(tasks),
            "tool_description_policy": "generated_name_only_v1",
            "verifiers_dropped": True,
            "server_configuration_dropped": True,
            "execution": "frozen_localagent_dense_selector_no_tool_execution",
        },
        "summary": summarize_scores(scores),
        "records": list(scores),
        "claim_boundary": "Out-of-domain name-only retrieval diagnostic; not an official EnterpriseOps-Gym task-success score, state-verifier result, or training artifact.",
    }
    # Keep the receipt independently auditable even though the source rows themselves remain
    # outside Git.  The digest covers the aggregate and per-task retrieval records, but never
    # serializes the dropped server configuration or verifier fields.
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
