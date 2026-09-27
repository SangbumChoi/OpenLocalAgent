#!/usr/bin/env python3
"""Evaluate warm/random checkpoints on the MCP-Persona tool-chain text projection."""

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
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _load_checkpoint(path: Path) -> tuple[LocalAgentLM, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config)
    model.load_state_dict(payload["state_dict"])
    return model, _checkpoint_tokenizer(payload)


def _group_metrics(model: LocalAgentLM, tokenizer: Any, rows: list[Conversation], max_seq_len: int, batch_size: int) -> dict[str, Any]:
    groups: dict[str, list[Conversation]] = defaultdict(list)
    for row in rows:
        groups[str(row.meta.get("first_server", "unknown"))].append(row)
    return {
        "overall": _evaluate_conversations(model, rows, tokenizer, max_seq_len=max_seq_len, batch_size=batch_size, device="cpu"),
        "by_first_server": {
            server: _evaluate_conversations(model, server_rows, tokenizer, max_seq_len=max_seq_len, batch_size=batch_size, device="cpu")
            for server, server_rows in sorted(groups.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = [Conversation.from_json(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    arms: dict[str, dict[str, Any]] = {}
    for name, checkpoint in (("warm", args.warm), ("random", args.random)):
        model, tokenizer = _load_checkpoint(checkpoint)
        arms[name] = {"checkpoint": _identity(checkpoint), "metrics": _group_metrics(model, tokenizer, rows, args.max_seq_len, args.batch_size)}
    warm_accuracy = arms["warm"]["metrics"]["overall"]["assistant_token_accuracy"]
    random_accuracy = arms["random"]["metrics"]["overall"]["assistant_token_accuracy"]
    report: dict[str, Any] = {
        "kind": "localagent_mcp_persona_tool_chain_projection_eval",
        "schema_version": 1,
        "dataset": {"data": _identity(args.data), "manifest": _identity(args.manifest), "manifest_sha256": manifest.get("manifest_sha256"), "rows": len(rows), "split": "test", "train_policy": "eval_only"},
        "protocol": {"max_seq_len": args.max_seq_len, "batch_size": args.batch_size, "target": "canonical_compact_json_tool_chain", "context_and_tools_excluded": True},
        "arms": arms,
        "comparison": {"warm_minus_random_assistant_token_accuracy": warm_accuracy - random_accuracy, "warm_beats_random": warm_accuracy > random_accuracy},
        "claim_boundary": "Teacher-forced tool-chain text projection only; no MCP simulator, persona state, ground-truth checkpoint judge, tool execution, or external side effect was run.",
    }
    report["receipt_self_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
