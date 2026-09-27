#!/usr/bin/env python
"""Run the low-rate-unfrozen arm of the synthetic MCP service-contract probe.

The probe uses only generated filesystem, GitHub, Notion, Playwright, and PostgreSQL contracts.
It does not train on MCPMark task text, state fixtures, verifiers, or servers.  The low-rate arm
keeps the public Mind2Web-adapted parent, updates the backbone and dispatch heads together, and
records tensor-group movement so it can be compared with the frozen transfer and matched-random
receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.routes import ROUTE_INDEX, RouteHead, route_of
from localagent.model.tokenizer import ASSISTANT, USER
from localagent.train.stage_data import probe_decisions
from scripts.train_mcp_service_probe import (
    _head_metrics,
    _identity,
    _load_checkpoint,
    _synthetic_rows,
    _tool_pool,
)
from scripts.train_mcp_service_probe_ablation import _combined_mcpmark_metrics


DEFAULT_TRANSFER_RECEIPT = Path(
    "docs/paper/results/raw/m93-mind2web-mcp-service-contract-v1.json"
)


def _relative_l2(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in before.items():
        if name not in after:
            continue
        left = value.detach().float()
        right = after[name].detach().float()
        numerator += float((right - left).pow(2).sum())
        denominator += float(left.pow(2).sum())
    return (numerator**0.5) / max(denominator**0.5, 1e-12)


def _batch_features(model, tokenizer, prompts: list[str], device: str) -> torch.Tensor:
    encoded = [
        tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")[-model.cfg.max_seq_len :]
        for prompt in prompts
    ]
    width = max(len(row) for row in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, row in enumerate(encoded):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    _, hidden = model(ids, return_hidden=True)
    return torch.stack([hidden[index, len(row) - 1] for index, row in enumerate(encoded)])


def _movement_groups(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    groups: dict[str, dict[str, torch.Tensor]] = {
        "embedding": {},
        "mixer": {},
        "ffn": {},
        "normalization": {},
    }
    for name, value in before.items():
        if name not in after:
            continue
        if name.startswith(("embed.", "in_proj.", "out_proj.")):
            groups["embedding"][name] = value
        elif ".attn." in name:
            groups["mixer"][name] = value
        elif ".ffn." in name:
            groups["ffn"][name] = value
        elif "norm." in name or name == "norm.weight":
            groups["normalization"][name] = value
    result = {"backbone": _relative_l2(before, after)}
    for group, values in groups.items():
        result[group] = _relative_l2(values, {name: after[name] for name in values})
    return result


def _state_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    return sum(
        float((after[name].detach().float() - value.detach().float()).norm())
        for name, value in before.items()
        if name in after
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mcpmark", type=Path)
    parser.add_argument("--transfer-report", type=Path, default=DEFAULT_TRANSFER_RECEIPT)
    parser.add_argument("--random-report", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite low-rate probe outputs")
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("steps and batch-size must be positive")

    parent, model, tokenizer = _load_checkpoint(args.init)
    model = model.to(args.device)
    parent_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.manual_seed(args.seed)
    rows = _synthetic_rows()
    decisions = probe_decisions(rows)
    tools = _tool_pool()
    route = RouteHead(model.cfg.d_model).to(args.device)
    route.load_state_dict(parent["route_head"])
    dense = DenseToolSelector(
        model.cfg.d_model,
        proj=int(parent["dense_selector"]["q_proj.weight"].shape[0]),
    ).to(args.device)
    dense.load_state_dict(parent["dense_selector"])
    embeddings = tool_embeddings(tools, device=args.device)
    names = [tool.name for tool in tools]
    name_index = {name: index for index, name in enumerate(names)}
    route_labels = [ROUTE_INDEX[route_of(item.ref_name)] for item in decisions]
    selector_labels = [name_index[item.ref_name] for item in decisions]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.backbone_lr},
            {"params": route.parameters(), "lr": args.head_lr},
            {"params": dense.parameters(), "lr": args.head_lr},
        ]
    )
    rng = random.Random(args.seed)
    model.train()
    route.train()
    dense.train()
    for step in range(args.steps):
        indices = [rng.randrange(len(decisions)) for _ in range(min(args.batch_size, len(decisions)))]
        prompts = [decisions[index].prompt for index in indices]
        features = _batch_features(model, tokenizer, prompts, args.device)
        route_target = torch.tensor([route_labels[index] for index in indices], device=args.device)
        selector_target = torch.tensor(
            [selector_labels[index] for index in indices], device=args.device
        )
        loss = F.cross_entropy(route(features), route_target) + F.cross_entropy(
            dense(features, embeddings), selector_target
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"[low-rate] step {step}/{args.steps} loss {loss.item():.4f}", flush=True)
    model.eval()
    route.eval()
    dense.eval()
    bound = BoundSelector(dense, tools)
    metrics = _head_metrics(model, tokenizer, route, bound, rows, tools)
    child = dict(parent)
    child.update(
        {
            "state_dict": model.state_dict(),
            "route_head": route.state_dict(),
            "dense_selector": dense.state_dict(),
            "stage": "mcp_service_contract_probe_lowrate",
            "mcp_service_probe": {
                "schema_version": 1,
                "training_rows": len(rows),
                "training_decisions": len(decisions),
                "services": sorted({row.meta["service"] for row in rows}),
                "source_text": "generated service/tool contracts only",
                "mcpmark_task_text_used": False,
                "backbone_initialization": "pretrained_low_rate_unfrozen",
                "random_seed": args.seed,
                "head_metrics": metrics,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report: dict[str, Any] = {
        "kind": "localagent_mcp_service_contract_lowrate_probe",
        "schema_version": 1,
        "parent": _identity(args.init),
        "child": _identity(args.output),
        "training": {
            "rows": len(rows),
            "decisions": len(decisions),
            "services": sorted({row.meta["service"] for row in rows}),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "random_seed": args.seed,
            "source_text": "generated service/tool contracts only",
            "mcpmark_task_text_used": False,
        },
        "head_metrics": metrics,
        "weight_delta": {
            "groups": _movement_groups(parent_state, model.state_dict()),
            "route_head": _state_delta(parent["route_head"], route.state_dict()),
            "dense_selector": _state_delta(parent["dense_selector"], dense.state_dict()),
        },
        "mcpmark": {},
        "transfer_reference": {
            "receipt": str(args.transfer_report),
            "receipt_sha256": hashlib.sha256(args.transfer_report.read_bytes()).hexdigest(),
            "backbone_initialization": "pretrained_frozen",
        },
        "claim_boundary": (
            "Low-rate-unfrozen synthetic service/tool-contract probe only; MCPMark fields are task-"
            "description routing proxy metrics, not live MCP execution, verifier success, pass@k, "
            "or an official leaderboard score."
        ),
    }
    if args.mcpmark is not None:
        transfer_receipt = json.loads(args.transfer_report.read_text(encoding="utf-8"))
        transfer_metrics = _combined_mcpmark_metrics(transfer_receipt)
        report["transfer_reference"].update(
            {f"combined_{metric}": value for metric, value in transfer_metrics.items()}
        )
        from localagent.eval.mcpmark_router import evaluate_mcpmark_router

        for suite in ("standard", "easy"):
            report["mcpmark"][suite] = evaluate_mcpmark_router(
                args.mcpmark, args.output, suite=suite, device=args.device
            )
        total_rows = sum(item["overall"]["rows"] for item in report["mcpmark"].values())
        lowrate_metrics = {
            metric: sum(
                item["overall"][metric] * item["overall"]["rows"]
                for item in report["mcpmark"].values()
            )
            / total_rows
            for metric in ("route_accuracy", "selector_top1", "selector_top3")
        }
        report["comparison"] = {
            "rows": total_rows,
            "transfer_pretrained_frozen": transfer_metrics,
            "lowrate_pretrained_unfrozen": lowrate_metrics,
            "lowrate_minus_transfer": {
                metric: lowrate_metrics[metric] - transfer_metrics[metric]
                for metric in transfer_metrics
            },
        }
        if args.random_report is not None:
            random_receipt = json.loads(args.random_report.read_text(encoding="utf-8"))
            random_metrics = _combined_mcpmark_metrics(random_receipt)
            report["random_reference"] = {
                "receipt": str(args.random_report),
                "receipt_sha256": hashlib.sha256(args.random_report.read_bytes()).hexdigest(),
                "backbone_initialization": "deterministic_random",
                **{f"combined_{metric}": value for metric, value in random_metrics.items()},
            }
            report["comparison"].update(
                {
                    "random_backbone": random_metrics,
                    "lowrate_minus_random": {
                        metric: lowrate_metrics[metric] - random_metrics[metric]
                        for metric in random_metrics
                    },
                }
            )
    report["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
