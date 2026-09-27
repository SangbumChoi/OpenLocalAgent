#!/usr/bin/env python3
"""Run a bounded free-run action probe over normalized ToolACE history conversations.

This evaluator uses the same catalog + history prompt contract as the WebGPU dispatch path, then
scores the first generated call at every assistant action turn.  It never dispatches a tool or
contacts an external service; the receipt is a deployment-shaped diagnostic, not an official
ToolACE/BFCL score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from localagent.agent.constrained import hybrid_decode
from localagent.agent.parser import extract_tool_calls
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.data.prompt_contract import render_function_catalog, schema_matches
from localagent.data.render import history_text
from localagent.data.schema import Conversation, Role, ToolSpec
from localagent.eval.tool_eval import match_calls
from localagent.model.tokenizer import ASSISTANT, BPE_EOS


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load_rows(path: Path, max_rows: int) -> list[Conversation]:
    rows: list[Conversation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(Conversation.from_json(line))
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"empty ToolACE action-history input: {path}")
    return rows


def _registry(rows: list[Conversation]) -> ToolRegistry:
    specs: dict[str, ToolSpec] = {}
    for row in rows:
        for spec in row.tools:
            specs.setdefault(spec.name, spec)
    registry = ToolRegistry()
    for spec in specs.values():
        registry.register(spec, lambda **kwargs: kwargs)
    return registry


def evaluate(checkpoint: Path, rows_path: Path, *, max_rows: int, device: str) -> dict[str, Any]:
    rows = _load_rows(rows_path, max_rows)
    agent = Agent.from_checkpoint(checkpoint, _registry(rows))
    counters = Counter()
    by_turn: defaultdict[str, Counter[str]] = defaultdict(Counter)
    predictions: list[dict[str, Any]] = []
    for row_index, conversation in enumerate(rows):
        catalog = render_function_catalog(conversation.tools) + BPE_EOS
        episode_ok = True
        episode_steps = 0
        for message_index, message in enumerate(conversation.messages):
            if message.role != Role.assistant or not message.tool_calls:
                continue
            episode_steps += 1
            grounding = history_text(conversation.messages[:message_index])
            prompt = catalog + grounding + ASSISTANT
            output = hybrid_decode(
                agent.model,
                agent.tokenizer,
                prompt,
                list(conversation.tools),
                device=device,
                selector=agent.selector,
                route_head=agent.route_head,
                ptr_head=agent.ptr_head,
                top_m=1,
                framed=True,
                grounding_prompt=grounding,
            )
            calls = extract_tool_calls(output)
            predicted = calls[0] if calls else None
            target = message.tool_calls[0]
            tool_exact = predicted is not None and predicted.name == target.name
            argument_exact = tool_exact and predicted.arguments == target.arguments
            schema_valid = predicted is not None and any(
                predicted.name == spec.name and schema_matches(predicted.arguments, spec.parameters)
                for spec in conversation.tools
            )
            step_ok = bool(predicted) and match_calls([predicted], [target])
            counters.update(
                steps=1,
                tool_exact=int(tool_exact),
                argument_exact=int(argument_exact),
                schema_valid=int(schema_valid),
                step_exact=int(step_ok),
            )
            by_turn[str(episode_steps)]["steps"] += 1
            by_turn[str(episode_steps)]["step_exact"] += int(step_ok)
            episode_ok = episode_ok and step_ok
            predictions.append(
                {
                    "row": row_index,
                    "message_index": message_index,
                    "target": {"name": target.name, "arguments": target.arguments},
                    "prediction": (
                        {"name": predicted.name, "arguments": predicted.arguments}
                        if predicted is not None
                        else None
                    ),
                    "tool_exact": bool(tool_exact),
                    "argument_exact": bool(argument_exact),
                    "schema_valid": bool(schema_valid),
                    "step_exact": bool(step_ok),
                }
            )
        if episode_steps:
            counters.update(episodes=1, episode_exact=int(episode_ok))
    steps = max(1, counters["steps"])
    episodes = max(1, counters["episodes"])
    return {
        "kind": "localagent_toolace_action_history_free_run_probe",
        "schema_version": 1,
        "checkpoint": _identity(checkpoint),
        "source": {
            "dataset": "Team-ACE/ToolACE",
            "url": "https://huggingface.co/datasets/Team-ACE/ToolACE",
            "revision": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
            "input": _identity(rows_path),
            "training_used": False,
        },
        "rows_requested": max_rows,
        "rows_evaluated": len(rows),
        "metrics": {
            "steps": counters["steps"],
            "episodes": counters["episodes"],
            "tool_exact_rate": counters["tool_exact"] / steps,
            "argument_exact_rate": counters["argument_exact"] / steps,
            "schema_valid_rate": counters["schema_valid"] / steps,
            "step_exact_rate": counters["step_exact"] / steps,
            "episode_exact_rate": counters["episode_exact"] / episodes,
            "by_turn": {
                key: {
                    "steps": value["steps"],
                    "step_exact_rate": value["step_exact"] / max(1, value["steps"]),
                }
                for key, value in sorted(by_turn.items(), key=lambda item: int(item[0]))
            },
        },
        "predictions": predictions,
        "claim_boundary": (
            "Bounded free-run ToolACE action-history probe with catalog-constrained generation; "
            "no tool dispatch, no external side effects, and no official ToolACE or BFCL score."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.rows, max_rows=args.max_rows, device=args.device)
    report["receipt_self_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
