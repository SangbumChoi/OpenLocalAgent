#!/usr/bin/env python3
"""Score a checkpoint's teacher-forced text prediction on AgentWorldBench projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.sft import _evaluate_conversations
from scripts.train_deployment_dispatch_repair import _checkpoint_tokenizer


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _manifest_identity(manifest: dict[str, Any]) -> str:
    """Use the manifest's own content hash, excluding the self-hash field when recomputing."""

    recorded = manifest.get("manifest_sha256")
    if isinstance(recorded, str) and len(recorded) == 64:
        return recorded
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(payload["state_dict"])
    tokenizer = _checkpoint_tokenizer(payload)
    rows: list[Conversation] = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(Conversation.from_json(line))
    groups: dict[str, list[Conversation]] = defaultdict(list)
    for row in rows:
        groups[str(row.meta.get("domain", "unknown"))].append(row)
    metrics = {
        "overall": _evaluate_conversations(
            model,
            rows,
            tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device="cpu",
        ),
        "by_domain": {
            domain: _evaluate_conversations(
                model,
                domain_rows,
                tokenizer,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                device="cpu",
            )
            for domain, domain_rows in sorted(groups.items())
        },
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "kind": "localagent_agentworldbench_text_projection_eval",
        "schema_version": 1,
        "checkpoint": _identity(args.checkpoint),
        "data": _identity(args.data),
        "manifest": {
            "path": str(args.manifest),
            "sha256": _manifest_identity(manifest),
            "dataset": manifest.get("dataset"),
            "source_revision": manifest.get("source_revision"),
            "split": manifest.get("split"),
            "train_policy": manifest.get("train_policy"),
        },
        "protocol": {
            "rows": len(rows),
            "domains": sorted(groups),
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "teacher_forced_last_observation_target": True,
        },
        "metrics": metrics,
        "claim_boundary": (
            "Teacher-forced text/world-model projection on the public AgentWorldBench test split. "
            "This is not an official AgentWorldBench judge score, action success, screenshot score, "
            "native Android/OS/browser/MCP execution, or training result; all rows remain eval-only."
        ),
    }
    report["receipt_self_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
