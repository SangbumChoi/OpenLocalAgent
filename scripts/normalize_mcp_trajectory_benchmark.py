#!/usr/bin/env python3
"""Normalize the public MCP trajectory benchmark into an internal agent-disjoint SFT split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role


DATASET = "obaydata/mcp-agent-trajectory-benchmark"
DATASET_URL = "https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark"
REVISION = "f4f449d65271abc1e4ccd5157d121a59a1dd38c4"
LICENSE = "Apache-2.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_target(tool_calls: list[dict[str, Any]]) -> str:
    return _canonical(
        {
            "tool_calls": [
                {
                    "name": str(call["function_name"]),
                    "arguments": call.get("arguments", {}),
                }
                for call in tool_calls
            ]
        }
    )


def _trajectory_rows(path: Path, split: str) -> list[Conversation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError(f"single-pass trajectory must contain steps: {path}")
    agent_name = path.parent.name
    rows: list[Conversation] = []
    steps = payload["steps"]
    for index, step in enumerate(steps):
        if step.get("source") != "user" or index + 1 >= len(steps):
            continue
        next_step = steps[index + 1]
        calls = next_step.get("tool_calls", [])
        if next_step.get("source") != "agent" or not isinstance(calls, list) or not calls:
            continue
        message = step.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        rows.append(
            Conversation(
                messages=[Message(Role.user, message), Message(Role.assistant, _tool_target(calls))],
                meta={
                    "dataset": DATASET,
                    "dataset_url": DATASET_URL,
                    "source_revision": REVISION,
                    "license": LICENSE,
                    "source_split": "train",
                    "split": split,
                    "train_policy": "train" if split == "internal_train" else "eval_only",
                    "source_family": "mcp_trajectory",
                    "source_agent": agent_name,
                    "parent_record_id": agent_name,
                    "trajectory_path": str(path),
                    "trajectory_step_id": int(next_step.get("step_id", index + 1)),
                    "domain": str(payload.get("agent", {}).get("name", agent_name)),
                    "tool_count": len(calls),
                    "observation_policy": "user_to_tool_calls;tool_outputs_and_reasoning_excluded",
                },
            )
        )
    return rows


def normalize(repo: Path) -> tuple[list[Conversation], list[Conversation], dict[str, Any]]:
    public_train = {
        str(json.loads(line)["agent_name"])
        for line in (repo / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    trajectory_paths = sorted(repo.glob("*/trajectory.json"))
    if len(trajectory_paths) != 38:
        raise ValueError(f"expected 38 single-pass trajectories, found {len(trajectory_paths)}")
    agents = [path.parent.name for path in trajectory_paths]
    if len(set(agents)) != len(agents):
        raise ValueError("single-pass trajectory agent names must be unique")
    if public_train != set(agents):
        raise ValueError("train.jsonl agent names do not match single-pass trajectory directories")
    eval_agents = set(agents[-8:])
    train_rows: list[Conversation] = []
    eval_rows: list[Conversation] = []
    per_agent: Counter[str] = Counter()
    for path in trajectory_paths:
        split = "internal_eval" if path.parent.name in eval_agents else "internal_train"
        rows = _trajectory_rows(path, split)
        per_agent[path.parent.name] = len(rows)
        (eval_rows if split == "internal_eval" else train_rows).extend(rows)
    multi_paths = sorted((repo / "multi_conv").glob("*/trajectory.json"))
    invalid_multi = 0
    for path in multi_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid_multi += 1
    selection = {
        "single_pass_trajectories": len(trajectory_paths),
        "internal_train_agents": sorted(set(agents) - eval_agents),
        "internal_eval_agents": sorted(eval_agents),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "rows_by_agent": dict(sorted(per_agent.items())),
        "multi_conv_trajectories_seen": len(multi_paths),
        "multi_conv_invalid_json": invalid_multi,
        "split_policy": "sorted_agent_name_first_30_train_last_8_eval;structural_internal_holdout_not_official",
    }
    return train_rows, eval_rows, selection


def _write(rows: list[Conversation], path: Path) -> str:
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (row.to_json() + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.train_output, args.eval_output, args.manifest)):
        raise SystemExit("refusing to overwrite normalization outputs")
    train_rows, eval_rows, selection = normalize(args.repo)
    train_sha = _write(train_rows, args.train_output)
    eval_sha = _write(eval_rows, args.eval_output)
    manifest: dict[str, Any] = {
        "kind": "localagent_mcp_trajectory_normalization_manifest",
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "source_revision": REVISION,
        "license": LICENSE,
        "selection": selection,
        "records": {
            "train": {"path": str(args.train_output), "rows": len(train_rows), "sha256": train_sha},
            "eval": {"path": str(args.eval_output), "rows": len(eval_rows), "sha256": eval_sha},
        },
        "claim_boundary": (
            "The public Hub split is labeled train but has no official held-out split. The 30/8 agent-disjoint "
            "partition is an internal structural holdout; reasoning, tool outputs, and MCP server execution "
            "are excluded from the Conversation projection."
        ),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
