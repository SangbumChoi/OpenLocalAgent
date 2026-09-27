#!/usr/bin/env python
"""Train a contamination-safe service/tool dispatch probe.

The probe uses only short, generated service/tool-contract prompts.  It does not read MCPMark
task descriptions, state fixtures, or verifiers.  The resulting checkpoint changes only the
route and dense-selector heads; the language-model backbone remains byte-for-byte frozen.  An
optional MCPMark proxy evaluation is run after training so the transfer claim is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
from localagent.agent.routes import train_route_head
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.schema import Conversation, Message, Role, ToolCall
from localagent.eval.mcpmark_router import evaluate_mcpmark_router
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import probe_decisions


SERVICE_PROMPTS: dict[str, tuple[str, ...]] = {
    "filesystem": (
        "Use the filesystem MCP server to read the project README file.",
        "Use the filesystem MCP server to list the files under the source directory.",
        "Use the filesystem MCP server to find Python files matching a test pattern.",
        "Use the filesystem MCP server to search the repository for a configuration key.",
        "Use the filesystem MCP server to create the requested output directory.",
        "Use the filesystem MCP server to write the requested report to disk.",
        "Use the filesystem MCP server to extract the downloaded archive.",
    ),
    "github": (
        "Use the GitHub MCP server to inspect the current repository status.",
        "Use the GitHub MCP server to show the pending diff for this change.",
        "Use the GitHub MCP server to read the repository's issue notes.",
        "Use the GitHub MCP server to apply the requested patch to the repository.",
        "Use the GitHub MCP server to commit the reviewed change.",
        "Use the GitHub MCP server to make an HTTP request to the repository API.",
    ),
    "notion": (
        "Use the Notion MCP server to create a project page with the supplied title and body.",
        "Use the Notion MCP server to append this note to the workspace page.",
        "Use the Notion MCP server to record the meeting decision in a new page.",
        "Use the Notion MCP server to write the requested status update.",
    ),
    "playwright": (
        "Use the Playwright browser MCP server to open the requested URL.",
        "Use the Playwright browser MCP server to click the named button.",
        "Use the Playwright browser MCP server to type the requested text.",
        "Use the Playwright browser MCP server to take a screenshot of the page.",
        "Use the Playwright browser MCP server to scroll down the page.",
        "Use the Playwright browser MCP server to press Enter.",
        "Use the Playwright browser MCP server to wait for the page to load.",
    ),
    "postgres": (
        "Use the PostgreSQL MCP server to run the requested SELECT query.",
        "Use the database MCP server to query the customer records.",
        "Use the PostgreSQL MCP server to inspect rows matching the supplied condition.",
        "Use the database MCP server to execute the provided SQL statement.",
    ),
}

SERVICE_TARGETS: dict[str, tuple[str, ...]] = {
    "filesystem": ("read_file", "list_dir", "find_files", "grep_search", "make_dir", "write_file", "unzip"),
    "github": ("git_status", "git_diff", "read_file", "apply_patch", "git_commit", "http_request"),
    "notion": ("notion_create_page", "notion_write", "notion_create_page", "notion_write"),
    "playwright": ("open_url", "click", "type_text", "screenshot", "scroll", "key_press", "wait"),
    "postgres": ("sql_query", "sql_query", "sql_query", "sql_query"),
}


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _tool_pool():
    return [*STANDARD_TOOLS, *mobile_tools(), *realistic_productivity_tools()]


def _synthetic_rows() -> list[Conversation]:
    rows: list[Conversation] = []
    for service, prompts in SERVICE_PROMPTS.items():
        targets = SERVICE_TARGETS[service]
        if len(prompts) != len(targets):
            raise AssertionError(f"synthetic contract mismatch for {service}")
        for index, (prompt, target) in enumerate(zip(prompts, targets, strict=True)):
            rows.append(
                Conversation(
                    messages=[
                        Message(role=Role.user, content=f"Request: {prompt}"),
                        Message(
                            role=Role.assistant,
                            tool_calls=[ToolCall(name=target, arguments={})],
                        ),
                    ],
                    meta={
                        "synthetic_probe": "mcp_service_contract_v1",
                        "service": service,
                        "index": index,
                    },
                )
            )
    return rows


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], LocalAgentLM, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    cfg = ModelConfig(**raw["cfg"])
    cfg.assert_within_budget()
    model = LocalAgentLM(cfg)
    model.load_state_dict(raw["state_dict"])
    metadata = raw.get("tokenizer") or {"kind": "byte"}
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")
    tokenizer = load_tokenizer(str(metadata.get("kind", "byte")), metadata.get("path"))
    return dict(raw), model, tokenizer


def _state_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, tensor in before.items():
        if name not in after:
            continue
        result[name] = float((after[name].detach().float() - tensor.detach().float()).norm().item())
    return result


def _head_metrics(model, tokenizer, route, selector, rows: list[Conversation], tools) -> dict[str, float | int]:
    decisions = probe_decisions(rows)
    if not decisions:
        return {"rows": 0, "route_accuracy": 0.0, "selector_top1": 0.0, "selector_top3": 0.0}
    from localagent.agent.tool_head import _feat
    from localagent.agent.routes import ROUTES, route_of

    features = torch.stack(
        [_feat(model, tokenizer, item.prompt, "cpu", framed=item.framed) for item in decisions]
    )
    with torch.no_grad():
        route_predictions = route(features).argmax(-1).tolist()
        rankings = selector.model(features, selector.embs).argsort(dim=-1, descending=True).tolist()
    names = selector.names
    route_correct = sum(ROUTES[pred] == route_of(item.ref_name) for pred, item in zip(route_predictions, decisions, strict=True))
    tool_rows = [item for item in decisions if item.kind == "tool" and item.ref_name in names]
    top1 = 0
    top3 = 0
    for item, ranking in zip(decisions, rankings, strict=True):
        if item.kind != "tool" or item.ref_name not in names:
            continue
        expected = names.index(item.ref_name)
        top1 += int(ranking[0] == expected)
        top3 += int(expected in ranking[:3])
    return {
        "rows": len(decisions),
        "tool_rows": len(tool_rows),
        "route_accuracy": route_correct / len(decisions),
        "selector_top1": top1 / max(1, len(tool_rows)),
        "selector_top3": top3 / max(1, len(tool_rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mcpmark", type=Path)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--head-steps", type=int, default=800)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite continuation outputs")
    if args.steps < 1 or args.head_steps < 1:
        raise SystemExit("steps must be positive")

    parent, model, tokenizer = _load_checkpoint(args.init)
    rows = _synthetic_rows()
    decisions = probe_decisions(rows)
    tools = _tool_pool()
    route_before = parent.get("route_head")
    selector_before = parent.get("dense_selector")
    if not isinstance(route_before, Mapping) or not isinstance(selector_before, Mapping):
        raise ValueError("checkpoint must contain route_head and dense_selector")
    route = train_route_head(model, decisions, tokenizer, steps=args.head_steps, batch_size=64, log=print)
    selector = train_dense_selector(
        model,
        decisions,
        tokenizer,
        tools,
        steps=args.steps,
        batch_size=64,
        proj=int(selector_before["q_proj.weight"].shape[0]),
        log=print,
    )
    bound = BoundSelector(selector, tools)
    metrics = _head_metrics(model, tokenizer, route, bound, rows, tools)

    child = dict(parent)
    child.update(
        {
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "stage": "mcp_service_contract_probe",
            "mcp_service_probe": {
                "schema_version": 1,
                "training_rows": len(rows),
                "training_decisions": len(decisions),
                "services": sorted(SERVICE_PROMPTS),
                "source_text": "generated service/tool contracts only",
                "mcpmark_task_text_used": False,
                "head_metrics": metrics,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)

    report: dict[str, Any] = {
        "kind": "localagent_mcp_service_contract_probe",
        "schema_version": 1,
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "training": {
            "rows": len(rows),
            "decisions": len(decisions),
            "services": sorted(SERVICE_PROMPTS),
            "route_steps": args.head_steps,
            "selector_steps": args.steps,
            "source_text": "generated service/tool contracts only",
            "mcpmark_task_text_used": False,
        },
        "head_metrics": metrics,
        "weight_delta": {
            "backbone": 0.0,
            "route_head": sum(_state_delta(route_before, route.state_dict()).values()),
            "dense_selector": sum(_state_delta(selector_before, selector.state_dict()).values()),
            "route_tensors": _state_delta(route_before, route.state_dict()),
            "selector_tensors": _state_delta(selector_before, selector.state_dict()),
        },
        "mcpmark": {},
        "claim_boundary": (
            "Synthetic service/tool-contract transfer probe only; MCPMark fields are populated "
            "only by the separate public task-description routing proxy, never by live servers, "
            "state fixtures, verifiers, or official execution scores."
        ),
    }
    if args.mcpmark is not None:
        for suite in ("standard", "easy"):
            report["mcpmark"][suite] = evaluate_mcpmark_router(
                args.mcpmark, args.output, suite=suite, device="cpu"
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
