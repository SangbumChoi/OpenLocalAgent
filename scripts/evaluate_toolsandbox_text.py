#!/usr/bin/env python3
"""Evaluate the constrained decoder on a public ToolSandbox source projection.

ToolSandbox is an executable, stateful benchmark.  The input used here is the repository's
static-AST projection, so this command deliberately does not import ToolSandbox, launch its
simulator, read verifiers, or call external services.  It does, however, run the actual
checkpoint-backed route/retrieval/grounding decoder against each row's candidate tool list and
checks the resulting call against the retained JSON schemas.  The result is therefore a useful
WebGPU dispatch diagnostic, not an official ToolSandbox score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from localagent.agent.constrained import hybrid_decode
from localagent.agent.parser import extract_tool_calls
from localagent.agent.retriever import ToolRetriever
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.prompt_contract import schema_matches
from localagent.data.schema import Conversation, ToolCall, ToolSpec


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
        raise ValueError(f"empty ToolSandbox projection: {path}")
    for index, row in enumerate(rows):
        if len(row.messages) < 2 or not row.messages[1].tool_calls:
            raise ValueError(f"ToolSandbox row {index} has no assistant target call")
        if not row.tools:
            raise ValueError(f"ToolSandbox row {index} has no candidate tool catalog")
        target = row.messages[1].tool_calls[0].name
        if target not in {tool.name for tool in row.tools}:
            raise ValueError(f"ToolSandbox row {index} target is absent from its candidate catalog")
    return rows


def _registry(rows: list[Conversation]) -> ToolRegistry:
    by_name: dict[str, ToolSpec] = {}
    for row in rows:
        for tool in row.tools:
            by_name.setdefault(tool.name, tool)
    registry = ToolRegistry()
    for tool in by_name.values():
        registry.register(tool, lambda **kwargs: kwargs)
    return registry


def _schema_valid(call: ToolCall | None, tools: list[ToolSpec]) -> bool:
    if call is None:
        return False
    spec = next((tool for tool in tools if tool.name == call.name), None)
    return spec is not None and schema_matches(call.arguments, spec.parameters)


def _decode(
    agent: Agent,
    row: Conversation,
    *,
    candidate_mode: str,
    device: str,
) -> ToolCall | None:
    if candidate_mode == "row_retriever":
        tools = list(row.tools)
        output = hybrid_decode(
            agent.model,
            agent.tokenizer,
            row.messages[0].content,
            tools,
            device=device,
            retriever=ToolRetriever(tools),
            route_head=agent.route_head,
            ptr_head=agent.ptr_head,
            top_m=1,
        )
    elif candidate_mode == "global_selector":
        tools = list(agent.catalog.values())
        output = hybrid_decode(
            agent.model,
            agent.tokenizer,
            row.messages[0].content,
            tools,
            device=device,
            selector=agent.selector,
            route_head=agent.route_head,
            ptr_head=agent.ptr_head,
            top_m=1,
        )
    else:  # pragma: no cover - argparse restricts this branch
        raise ValueError(f"unsupported candidate mode: {candidate_mode}")
    calls = extract_tool_calls(output)
    return calls[0] if calls else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _score_rows(
    agent: Agent,
    rows: list[Conversation],
    *,
    candidate_mode: str,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counters = {"route": 0, "tool_exact": 0, "arguments_exact": 0, "schema_valid": 0}
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "route": 0, "tool_exact": 0, "arguments_exact": 0, "schema_valid": 0}
    )
    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target = row.messages[1].tool_calls[0]
        predicted = _decode(agent, row, candidate_mode=candidate_mode, device=device)
        route = predicted is not None
        tool_exact = route and predicted.name == target.name
        arguments_exact = tool_exact and predicted.arguments == target.arguments
        schema_valid = _schema_valid(predicted, list(row.tools))
        for key, value in {
            "route": route,
            "tool_exact": tool_exact,
            "arguments_exact": arguments_exact,
            "schema_valid": schema_valid,
        }.items():
            counters[key] += int(value)
        row_categories = [str(value) for value in row.meta.get("categories", [])] or ["UNCATEGORIZED"]
        for category in row_categories:
            category_counts = categories[category]
            category_counts["rows"] += 1
            category_counts["route"] += int(route)
            category_counts["tool_exact"] += int(tool_exact)
            category_counts["arguments_exact"] += int(arguments_exact)
            category_counts["schema_valid"] += int(schema_valid)
        prediction_rows.append(
            {
                "row_index": index,
                "task_id": row.meta.get("parent_record_id"),
                "categories": row_categories,
                "ground_truth": {"name": target.name, "arguments": target.arguments},
                "prediction": (
                    {"name": predicted.name, "arguments": predicted.arguments}
                    if predicted is not None
                    else None
                ),
                "route": route,
                "tool_exact": tool_exact,
                "arguments_exact": arguments_exact,
                "schema_valid": schema_valid,
            }
        )
    total = len(rows)
    overall = {
        "rows": total,
        "route_rate": _rate(counters["route"], total),
        "tool_exact_rate": _rate(counters["tool_exact"], total),
        "arguments_exact_rate": _rate(counters["arguments_exact"], total),
        "schema_valid_rate": _rate(counters["schema_valid"], total),
    }
    by_category = {
        category: {
            "rows": counts["rows"],
            "route_rate": _rate(counts["route"], counts["rows"]),
            "tool_exact_rate": _rate(counts["tool_exact"], counts["rows"]),
            "arguments_exact_rate": _rate(counts["arguments_exact"], counts["rows"]),
            "schema_valid_rate": _rate(counts["schema_valid"], counts["rows"]),
        }
        for category, counts in sorted(categories.items())
    }
    return {"overall": overall, "by_category": by_category}, prediction_rows


def evaluate(
    checkpoint: Path,
    eval_data: Path,
    predictions_path: Path,
    *,
    report_path: Path,
    candidate_mode: str = "row_retriever",
    device: str = "cpu",
) -> dict[str, Any]:
    rows = _load_rows(eval_data)
    registry = _registry(rows)
    agent = Agent.from_checkpoint(checkpoint, registry)
    metrics, prediction_rows = _score_rows(
        agent,
        rows,
        candidate_mode=candidate_mode,
        device=device,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    report = {
        "kind": "localagent_toolsandbox_text_projection_eval",
        "schema_version": 1,
        "dataset": {
            "name": "apple/ToolSandbox",
            "source_url": "https://github.com/apple/ToolSandbox",
            "revision": str(rows[0].meta.get("provenance", {}).get("revision", "unknown")),
            "projection": _identity(eval_data),
            "rows": len(rows),
            "candidate_tools": len({tool.name for row in rows for tool in row.tools}),
            "official_split_verified": False,
            "simulator_executed": False,
            "verifiers_executed": False,
            "external_api_called": False,
        },
        "checkpoint": _identity(checkpoint),
        "evaluator": {
            "script": _identity(Path(__file__).resolve()),
            "candidate_mode": candidate_mode,
            "decoder": "localagent.agent.constrained.hybrid_decode",
            "schema_checker": "localagent.data.prompt_contract.schema_matches",
        },
        "metrics": metrics,
        "predictions": _identity(predictions_path),
        "claim_boundary": (
            "Offline ToolSandbox text-observation/action projection only. The checkpoint-backed "
            "decoder was run against each retained candidate list and schema, but the upstream "
            "ToolSandbox simulator, user simulator, milestone verifiers, external services, and "
            "official benchmark split were not executed. This is not an official ToolSandbox "
            "score, MCPMark score, or native WebGPU capability result."
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
    parser.add_argument(
        "--candidate-mode",
        choices=("row_retriever", "global_selector"),
        default="row_retriever",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.predictions.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite ToolSandbox evaluation outputs")
    report = evaluate(
        args.checkpoint,
        args.eval_data,
        args.predictions,
        report_path=args.report,
        candidate_mode=args.candidate_mode,
        device=args.device,
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
