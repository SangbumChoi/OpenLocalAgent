#!/usr/bin/env python3
"""Evaluate a checkpoint on a bounded public xLAM-derived function-call shard.

The official Salesforce/xLAM dataset is gated. This evaluator accepts a hash-pinned JSONL export
from a public derivative mirror, preserves the original query/tools/answers fields, and runs the
checkpoint through the real constrained decoder. It is deliberately first-call focused: the
10.5M deployment model emits one bounded action, while xLAM rows may contain multiple calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from localagent.agent.constrained import hybrid_decode
from localagent.agent.parser import extract_tool_calls
from localagent.agent.retriever import ToolRetriever
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.prompt_contract import schema_matches
from localagent.data.public_agent import _xlam_tool
from localagent.data.schema import ToolCall, ToolSpec


ALL_MODES = (
    "row_retriever",
    "global_retriever",
    "runtime_retriever_selector",
    "global_selector",
)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _json_field(value: Any, *, label: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not JSON") from error
    return value


def _load_rows(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"row {line_number} must be an object")
        query = raw.get("query")
        tools_raw = _json_field(raw.get("tools"), label=f"row {line_number}.tools")
        answers_raw = _json_field(raw.get("answers"), label=f"row {line_number}.answers")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"row {line_number}.query must be non-empty text")
        if not isinstance(tools_raw, list) or not tools_raw:
            raise ValueError(f"row {line_number}.tools must be a non-empty list")
        if not isinstance(answers_raw, list) or not answers_raw:
            raise ValueError(f"row {line_number}.answers must be a non-empty list")
        tools = tuple(
            sorted(
                (
                    _xlam_tool(tool, label=f"row {line_number}.tools[{index}]")
                    for index, tool in enumerate(tools_raw)
                ),
                key=lambda tool: tool.name,
            )
        )
        answers: list[ToolCall] = []
        for index, answer in enumerate(answers_raw):
            if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
                raise ValueError(f"row {line_number}.answers[{index}] is malformed")
            arguments = _json_field(
                answer.get("arguments", {}),
                label=f"row {line_number}.answers[{index}].arguments",
            )
            if not isinstance(arguments, dict):
                raise ValueError(f"row {line_number}.answers[{index}].arguments must be an object")
            answers.append(ToolCall(name=answer["name"], arguments=arguments))
        names = {tool.name for tool in tools}
        if any(call.name not in names for call in answers):
            raise ValueError(f"row {line_number} answer tool is absent from candidates")
        rows.append(
            {"id": raw.get("id", line_number - 1), "query": query, "tools": tools, "answers": answers}
        )
        if max_rows and len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError(f"empty xLAM-derived shard: {path}")
    return rows


def _registry(rows: list[dict[str, Any]]) -> ToolRegistry:
    tools: dict[str, ToolSpec] = {}
    for row in rows:
        for tool in row["tools"]:
            tools.setdefault(tool.name, tool)
    registry = ToolRegistry()
    for tool in tools.values():
        registry.register(tool, lambda **kwargs: kwargs)
    return registry


def _decode(agent: Agent, row: dict[str, Any], mode: str, device: str) -> ToolCall | None:
    tools = list(row["tools"])
    if mode == "row_retriever":
        output = hybrid_decode(
            agent.model,
            agent.tokenizer,
            row["query"],
            tools,
            device=device,
            retriever=ToolRetriever(tools),
            route_head=agent.route_head,
            ptr_head=agent.ptr_head,
            top_m=1,
        )
    elif mode == "global_retriever":
        output = hybrid_decode(
            agent.model,
            agent.tokenizer,
            row["query"],
            list(agent.catalog.values()),
            device=device,
            retriever=agent.retriever,
            route_head=agent.route_head,
            ptr_head=agent.ptr_head,
            k=agent.retrieve_k,
            top_m=1,
        )
    elif mode == "runtime_retriever_selector":
        # Mirror Agent.chat: retrieve a bounded catalog first, then let the learned two-tower
        # selector rank only those candidates. This prevents a full-catalog selector from
        # silently defeating the runtime's O(top-k) contract.
        tools = agent._select_specs(row["query"])
        output = hybrid_decode(
            agent.model,
            agent.tokenizer,
            row["query"],
            tools,
            device=device,
            selector=agent.selector,
            route_head=agent.route_head,
            ptr_head=agent.ptr_head,
            top_m=1,
        )
    elif mode == "global_selector":
        output = hybrid_decode(
            agent.model,
            agent.tokenizer,
            row["query"],
            list(agent.catalog.values()),
            device=device,
            selector=agent.selector,
            route_head=agent.route_head,
            ptr_head=agent.ptr_head,
            top_m=1,
        )
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown candidate mode {mode}")
    calls = extract_tool_calls(output)
    return calls[0] if calls else None


def _schema_valid(call: ToolCall | None, tools: tuple[ToolSpec, ...]) -> bool:
    if call is None:
        return False
    spec = next((tool for tool in tools if tool.name == call.name), None)
    return spec is not None and schema_matches(call.arguments, spec.parameters)


def evaluate(
    checkpoint: Path,
    rows_path: Path,
    output: Path,
    *,
    max_rows: int,
    device: str,
    modes: Sequence[str] = ALL_MODES,
) -> dict[str, Any]:
    selected_modes = tuple(modes)
    if not selected_modes:
        raise ValueError("at least one candidate mode is required")
    unknown_modes = sorted(set(selected_modes) - set(ALL_MODES))
    if unknown_modes:
        raise ValueError(f"unknown candidate mode(s): {unknown_modes}")
    rows = _load_rows(rows_path, max_rows=max_rows)
    registry = _registry(rows)
    agent = Agent.from_checkpoint(checkpoint, registry)
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        counters = defaultdict(int)
        by_length: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        predictions: list[dict[str, Any]] = []
        for row in rows:
            target = row["answers"][0]
            predicted = _decode(agent, row, mode, device)
            exact_tool = predicted is not None and predicted.name == target.name
            exact_args = exact_tool and predicted.arguments == target.arguments
            valid = _schema_valid(predicted, row["tools"])
            counters["rows"] += 1
            counters["route"] += int(predicted is not None)
            counters["tool_exact"] += int(exact_tool)
            counters["arguments_exact"] += int(exact_args)
            counters["schema_valid"] += int(valid)
            length = str(len(row["answers"]))
            bucket = by_length[length]
            bucket["rows"] += 1
            bucket["tool_exact"] += int(exact_tool)
            bucket["arguments_exact"] += int(exact_args)
            predictions.append(
                {
                    "id": row["id"],
                    "target": {"name": target.name, "arguments": target.arguments},
                    "prediction": (
                        {"name": predicted.name, "arguments": predicted.arguments}
                        if predicted is not None
                        else None
                    ),
                    "tool_exact": exact_tool,
                    "arguments_exact": exact_args,
                    "schema_valid": valid,
                }
            )
        n = counters["rows"]
        by_mode[mode] = {
            "rows": n,
            "route_rate": counters["route"] / n,
            "first_tool_exact_rate": counters["tool_exact"] / n,
            "first_arguments_exact_rate": counters["arguments_exact"] / n,
            "schema_valid_rate": counters["schema_valid"] / n,
            "by_answer_count": {
                key: {
                    "rows": value["rows"],
                    "first_tool_exact_rate": value["tool_exact"] / value["rows"],
                    "first_arguments_exact_rate": value["arguments_exact"] / value["rows"],
                }
                for key, value in sorted(by_length.items(), key=lambda item: int(item[0]))
            },
            "predictions": predictions,
        }
    report = {
        "kind": "localagent_xlam_derived_function_calling_eval",
        "schema_version": 1,
        "source": {
            "derived_dataset": "product-science/xlam-function-calling-60k-raw",
            "derived_url": "https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw",
            "original_dataset": "Salesforce/xlam-function-calling-60k",
            "original_url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
            "original_access": "gated_and_not_authenticated_in_this_environment",
            "split": "test",
            "source_file": _identity(rows_path),
            "official_original_split_verified": False,
            "derived_license": "Apache-2.0",
            "training_used": False,
        },
        "checkpoint": _identity(checkpoint),
        "rows": len(rows),
        "modes": by_mode,
        "candidate_modes": list(selected_modes),
        "candidate_policy": {
            "runtime_retriever_selector": "Agent.chat retrieval followed by selector over retrieved candidates",
            "retrieve_k": agent.retrieve_k,
        },
        "claim_boundary": (
            "Bounded first-call evaluation on a public xLAM-derived test shard. xLAM answers may "
            "contain multiple calls; this 10.5M decoder is scored on the first call only. The "
            "official gated Salesforce source split, multi-call exactness, live APIs, and external "
            "side effects were not used."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=ALL_MODES,
        help="candidate mode to run; repeat for multiple modes (default: all modes)",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    report = evaluate(
        args.checkpoint,
        args.rows,
        args.output,
        max_rows=args.max_rows,
        device=args.device,
        modes=tuple(args.modes or ALL_MODES),
    )
    print(
        json.dumps(
            {
                mode: {key: value for key, value in metrics.items() if key != "predictions"}
                for mode, metrics in report["modes"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
