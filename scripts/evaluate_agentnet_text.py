#!/usr/bin/env python3
"""Evaluate a checkpoint on the retained text-only AgentNet action projection.

The public AgentNet trajectories are image-grounded.  This command intentionally consumes only
the already-normalized ``Conversation`` rows produced by :mod:`ingest_agentnet_text`: task text,
the bounded textual observation, and the retained action labels.  It does not open screenshots,
launch an OS, or treat the result as the official AgentNetBench leaderboard score.  The generated
JSONL uses the upstream action shape so the repository's strict AgentNet scorer can report
coordinate/text/keyboard/scroll metrics with exact parent coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from localagent.agent.constrained import hybrid_decode
from localagent.agent.parser import extract_tool_calls
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.agentnet import _tool
from localagent.data.schema import Conversation, ToolSpec
from localagent.eval.agentnet_results import aggregate_agentnet_results

_MODEL_TO_ACTION = {
    "agentnet_click": "click",
    "agentnet_double_click": "doubleClick",
    "agentnet_right_click": "rightClick",
    "agentnet_middle_click": "middleClick",
    "agentnet_move_cursor": "moveTo",
    "agentnet_drag": "dragTo",
    "agentnet_scroll": "scroll",
    "agentnet_hscroll": "hscroll",
    "agentnet_type_text": "write",
    "agentnet_key_press": "press",
    "agentnet_hotkey": "hotkey",
    "agentnet_triple_click": "tripleClick",
    "agentnet_wait": "wait",
    "agentnet_terminate": "terminate",
}
_SOURCE_TO_ACTION = {
    "click": "click",
    "double_click": "doubleClick",
    "move_cursor": "moveTo",
    "drag": "dragTo",
    "scroll": "scroll",
    "type_text": "write",
    "key_press": "press",
    "wait": "wait",
}
_ACTION_NAMES = (
    "click",
    "doubleClick",
    "rightClick",
    "middleClick",
    "moveTo",
    "dragTo",
    "scroll",
    "hscroll",
    "write",
    "press",
    "hotkey",
    "tripleClick",
    "wait",
    "terminate",
)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load_rows(path: Path) -> list[Conversation]:
    rows = [
        Conversation.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty AgentNet projection: {path}")
    for row in rows:
        if not row.meta.get("parent_record_id"):
            raise ValueError("AgentNet projection row is missing parent_record_id")
    return rows


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for action_name in _ACTION_NAMES:
        raw = _tool(action_name)
        registry.register(
            ToolSpec(
                name=raw["name"],
                description=raw["description"],
                parameters=raw["parameters"],
            ),
            lambda **kwargs: kwargs,
        )
    return registry


def _xy(target: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for part in str(target).split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key in {"x", "y"}:
            values[key] = float(value)
    if set(values) != {"x", "y"}:
        raise ValueError(f"coordinate target is incomplete: {target!r}")
    return values


def _source_action(row: Conversation) -> str:
    code = str(row.meta.get("action_code", ""))
    for source, action in (
        ("tripleClick", "tripleClick"),
        ("rightClick", "rightClick"),
        ("middleClick", "middleClick"),
        ("doubleClick", "doubleClick"),
        ("hotkey", "hotkey"),
        ("press", "press"),
    ):
        if source in code:
            return action
    name = row.messages[1].tool_calls[0].name
    return _SOURCE_TO_ACTION.get(name, name)


def _argument_mapping(raw: Any) -> dict[str, Any]:
    """Normalize AgentNet's mapping and legacy ``[name, mapping]`` argument forms."""

    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[1], Mapping):
        return dict(raw[1])
    raise ValueError(f"unsupported AgentNet argument shape: {raw!r}")


def _ground_truth(row: Conversation) -> dict[str, Any]:
    call = row.messages[1].tool_calls[0]
    action = _source_action(row)
    args = _argument_mapping(call.arguments)
    if action in {
        "click",
        "doubleClick",
        "rightClick",
        "middleClick",
        "moveTo",
        "dragTo",
        "tripleClick",
    }:
        target = args.get("dest", args.get("target", ""))
        params: dict[str, Any] = {"position": _xy(target)}
    elif action == "write":
        params = {"text": str(args.get("text", ""))}
    elif action in {"press", "hotkey"}:
        key = str(args.get("key", ""))
        params = {"keys": key.split("+") if action == "hotkey" else [key]}
    elif action in {"scroll", "hscroll"}:
        params = {"amount": 1 if args.get("direction") == "up" else -1}
    elif action == "wait":
        params = {"seconds": int(args.get("seconds", 0))}
    elif action == "terminate":
        params = {"status": str(args.get("status", "success"))}
    else:
        raise ValueError(f"unsupported projected action {action!r}")
    return {"type": action, "params": params}


def _prediction(call_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    action = _MODEL_TO_ACTION[call_name]
    if action in {
        "click",
        "doubleClick",
        "rightClick",
        "middleClick",
        "moveTo",
        "dragTo",
        "tripleClick",
    }:
        params: dict[str, Any] = {
            "position": {"x": arguments.get("x"), "y": arguments.get("y")}
        }
    elif action == "write":
        params = {"text": arguments.get("text", "")}
    elif action in {"press", "hotkey"}:
        keys = arguments.get("keys", arguments.get("key", []))
        params = {"keys": keys if isinstance(keys, list) else [keys]}
    elif action in {"scroll", "hscroll"}:
        params = {"amount": arguments.get("clicks", arguments.get("direction"))}
    elif action == "wait":
        params = {"seconds": arguments.get("seconds", 0)}
    else:
        params = {"status": arguments.get("status", "success")}
    return {"type": action, "params": params}


def evaluate(
    checkpoint: Path,
    eval_data: Path,
    predictions_path: Path,
    *,
    report_path: Path,
    device: str = "cpu",
    max_parents: int = 0,
    max_rows: int = 0,
) -> dict[str, Any]:
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    rows = _load_rows(eval_data)
    grouped: dict[str, list[Conversation]] = defaultdict(list)
    for row in rows:
        grouped[str(row.meta["parent_record_id"])].append(row)
    for parent_rows in grouped.values():
        parent_rows.sort(key=lambda row: int(row.meta.get("step_index", 0)))
    parent_items = sorted(grouped.items())
    if max_parents:
        if max_parents < 1:
            raise ValueError("max_parents must be positive")
        parent_items = parent_items[:max_parents]
    if max_rows:
        bounded_items: list[tuple[str, list[Conversation]]] = []
        remaining = max_rows
        for parent_id, parent_rows in parent_items:
            if remaining <= 0:
                break
            selected = parent_rows[:remaining]
            bounded_items.append((parent_id, selected))
            remaining -= len(selected)
        parent_items = bounded_items

    agent = Agent.from_checkpoint(checkpoint, _registry())
    prediction_records: list[dict[str, Any]] = []
    ground_truth_records: list[dict[str, Any]] = []
    for parent_id, parent_rows in parent_items:
        predicted_actions: list[dict[str, Any]] = []
        ground_truth_actions = [_ground_truth(row) for row in parent_rows]
        for row in parent_rows:
            prompt = row.messages[0].content
            output = hybrid_decode(
                agent.model,
                agent.tokenizer,
                prompt,
                list(agent.catalog.values()),
                device=device,
                selector=agent.selector,
                route_head=agent.route_head,
                ptr_head=agent.ptr_head,
                top_m=1,
            )
            calls = extract_tool_calls(output)
            if calls and calls[0].name in _MODEL_TO_ACTION:
                predicted_actions.append(_prediction(calls[0].name, calls[0].arguments))
        ground_truth_records.append(
            {"task_id": parent_id, "platform": "Ubuntu", "ground_truth_actions": ground_truth_actions}
        )
        prediction_records.append(
            {"task_id": parent_id, "platform": "Ubuntu", "predicted_actions": predicted_actions}
        )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path = predictions_path.with_name(predictions_path.stem + ".ground_truth.jsonl")
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_records), encoding="utf-8"
    )
    ground_truth_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ground_truth_records), encoding="utf-8"
    )
    expected_ids = [row["task_id"] for row in ground_truth_records]
    aggregate = aggregate_agentnet_results(
        ground_truth_path,
        predictions_path,
        expected_ids=expected_ids,
        source_revision=str(rows[0].meta.get("source_revision", "unknown")),
    )
    report = {
        "kind": "localagent_agentnet_text_projection_eval",
        "schema_version": 1,
        "checkpoint": _identity(checkpoint),
        "projection": _identity(eval_data),
        "rows": {"projected_actions": len(rows), "parents": len(parent_items)},
        "bounds": {"max_parents": max_parents, "max_rows": max_rows},
        "predictions": _identity(predictions_path),
        "ground_truth": _identity(ground_truth_path),
        "overall": aggregate["overall"],
        "by_platform": aggregate["by_platform"],
        "completeness": aggregate["completeness"],
        "claim_boundary": (
            "Offline AgentNet text-observation/action projection only. Screenshots, OS state, and "
            "the upstream desktop runtime were not used; dropped source actions (including the "
            "termination markers removed by the projection) are outside this score. This is not "
            "an official AgentNetBench leaderboard result or native desktop success."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if args.predictions.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite AgentNet evaluation outputs")
    report = evaluate(
        args.checkpoint,
        args.eval_data,
        args.predictions,
        report_path=args.report,
        device=args.device,
        max_parents=args.max_parents,
        max_rows=args.max_rows,
    )
    print(json.dumps({"overall": report["overall"], "completeness": report["completeness"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
