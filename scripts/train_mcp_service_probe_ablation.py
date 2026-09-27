#!/usr/bin/env python
"""Run a matched random-backbone control for the synthetic MCP service-contract probe.

The default transfer arm is the published ``m38`` probe: it freezes a pretrained LocalAgent
backbone and re-trains only route/selector heads on generated Notion, browser, filesystem, GitHub,
and database contracts.  Use ``--transfer-report`` to bind a different matched transfer receipt.
This control keeps the same rows, tokenizer, architecture, and optimization budget but starts the
backbone from a fresh deterministic seed.  MCPMark descriptions are used only for the separate
routing proxy; no task text, state fixture, verifier, or MCP server enters training.
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
from localagent.agent.routes import train_route_head
from localagent.eval.mcpmark_router import evaluate_mcpmark_router
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from scripts.train_mcp_service_probe import (
    SERVICE_PROMPTS,
    _head_metrics,
    _identity,
    _synthetic_rows,
    _tool_pool,
)
from localagent.train.stage_data import probe_decisions


MCPMARK_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"
TRANSFER_RECEIPT = "docs/paper/results/raw/m38-mcp-service-contract-probe-v1.json"


def _load_parent(path: Path) -> tuple[dict[str, Any], ModelConfig, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    cfg = ModelConfig(**raw["cfg"])
    cfg.assert_within_budget()
    metadata = raw.get("tokenizer") or {"kind": "byte"}
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")
    tokenizer = load_tokenizer(str(metadata.get("kind", "byte")), metadata.get("path"))
    return dict(raw), cfg, tokenizer


def _receipt_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_mcpmark_metrics(receipt: Mapping[str, Any]) -> dict[str, float]:
    suites = receipt.get("mcpmark")
    if not isinstance(suites, Mapping):
        raise ValueError("transfer receipt must contain mcpmark suite metrics")
    overall = [suites[name]["overall"] for name in ("standard", "easy")]
    rows = sum(int(item["rows"]) for item in overall)
    if rows <= 0:
        raise ValueError("transfer receipt must contain positive MCPMark rows")
    return {
        metric: sum(float(item[metric]) * int(item["rows"]) for item in overall) / rows
        for metric in ("route_accuracy", "selector_top1", "selector_top3")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mcpmark", type=Path)
    parser.add_argument(
        "--transfer-report",
        type=Path,
        default=Path(TRANSFER_RECEIPT),
        help="receipt for the matched pretrained transfer arm",
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--head-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite ablation outputs")
    if args.steps < 1 or args.head_steps < 1:
        raise SystemExit("steps must be positive")

    parent, cfg, tokenizer = _load_parent(args.init)
    torch.manual_seed(args.seed)
    model = LocalAgentLM(cfg)
    rows = _synthetic_rows()
    decisions = probe_decisions(rows)
    tools = _tool_pool()
    route = train_route_head(model, decisions, tokenizer, steps=args.head_steps, batch_size=64, log=print)
    selector = train_dense_selector(
        model,
        decisions,
        tokenizer,
        tools,
        steps=args.steps,
        batch_size=64,
        proj=int(parent["dense_selector"]["q_proj.weight"].shape[0]),
        log=print,
    )
    bound = BoundSelector(selector, tools)
    metrics = _head_metrics(model, tokenizer, route, bound, rows, tools)

    child = dict(parent)
    child.update(
        {
            "state_dict": model.state_dict(),
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "stage": "mcp_service_contract_probe_no_transfer",
            "mcp_service_probe": {
                "schema_version": 1,
                "training_rows": len(rows),
                "training_decisions": len(decisions),
                "services": sorted(SERVICE_PROMPTS),
                "source_text": "generated service/tool contracts only",
                "mcpmark_task_text_used": False,
                "backbone_initialization": "deterministic_random",
                "random_seed": args.seed,
                "head_metrics": metrics,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)

    report: dict[str, Any] = {
        "kind": "localagent_mcp_service_contract_no_transfer_probe",
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
            "backbone_initialization": "deterministic_random",
            "random_seed": args.seed,
        },
        "head_metrics": metrics,
        "mcpmark": {},
        "transfer_reference": {
            "receipt": str(args.transfer_report),
            "receipt_sha256": _receipt_sha256(args.transfer_report),
            "backbone_initialization": "pretrained_frozen",
        },
        "claim_boundary": (
            "Matched random-backbone control for a synthetic service/tool-contract probe; MCPMark "
            "values are task-description routing proxy metrics only, not live MCP execution, "
            "verifier success, pass@k, or an official leaderboard score."
        ),
    }
    if args.mcpmark is not None:
        transfer_receipt = json.loads(args.transfer_report.read_text(encoding="utf-8"))
        transfer_metrics = _combined_mcpmark_metrics(transfer_receipt)
        report["transfer_reference"].update(
            {
                f"combined_{metric}": value
                for metric, value in transfer_metrics.items()
            }
        )
        for suite in ("standard", "easy"):
            report["mcpmark"][suite] = evaluate_mcpmark_router(
                args.mcpmark, args.output, suite=suite, device="cpu"
            )
        total_rows = sum(item["overall"]["rows"] for item in report["mcpmark"].values())
        random_metrics = {
            metric: sum(
                item["overall"][metric] * item["overall"]["rows"]
                for item in report["mcpmark"].values()
            )
            / total_rows
            for metric in ("route_accuracy", "selector_top1", "selector_top3")
        }
        transfer_metrics = {
            metric: report["transfer_reference"][f"combined_{metric}"]
            for metric in ("route_accuracy", "selector_top1", "selector_top3")
        }
        report["comparison"] = {
            "rows": total_rows,
            "transfer_pretrained_frozen": transfer_metrics,
            "random_backbone": random_metrics,
            "random_minus_transfer": {
                metric: random_metrics[metric] - transfer_metrics[metric]
                for metric in transfer_metrics
            },
        }
    report["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
