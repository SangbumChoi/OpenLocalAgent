#!/usr/bin/env python3
"""Train a dynamic MCP tool selector on redacted trajectories and compare model bodies.

The selector is a frozen-feature two-tower probe.  It sees the union of tool schemas from the
train/eval conversations, while tool results and assistant free text remain redacted in the input
rows.  The warm arm loads the WebGPU checkpoint; the matched control uses the same architecture,
tokenizer, and seed-controlled random backbone.  This is a routing/transfer diagnostic only: it
does not start MCP servers, invoke a browser, or claim official MCPMark task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings, train_dense_selector
from localagent.agent.tool_head import _feat
from localagent.data.schema import Conversation, ToolSpec
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import ProbeDecision, probe_decisions


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_rows(paths: list[Path]) -> list[Conversation]:
    rows: list[Conversation] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(Conversation.from_json(line))
            except Exception as error:  # pragma: no cover - diagnostic context
                raise ValueError(f"invalid Conversation at {path}:{line_number}") from error
    if not rows:
        raise ValueError("no conversations supplied")
    return rows


def _tool_catalog(rows: list[Conversation]) -> list[ToolSpec]:
    by_name: dict[str, ToolSpec] = {}
    for row in rows:
        for tool in row.tools:
            by_name.setdefault(tool.name, tool)
    if not by_name:
        raise ValueError("conversation rows contain no tools")
    return [by_name[name] for name in sorted(by_name)]


def _checkpoint_tokenizer(parent: dict[str, Any]):
    metadata = parent.get("tokenizer") or {"kind": "byte"}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")
    return load_tokenizer(str(metadata.get("kind", "byte")), metadata.get("path"))


def _model(parent: dict[str, Any], *, warm: bool, seed: int) -> LocalAgentLM:
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    if warm:
        model = LocalAgentLM(config)
        model.load_state_dict(parent["state_dict"])
    else:
        torch.manual_seed(seed)
        model = LocalAgentLM(config)
    model.eval()
    return model


def _tool_rows(decisions: list[ProbeDecision], names: set[str]) -> list[ProbeDecision]:
    return [decision for decision in decisions if decision.kind == "tool" and decision.ref_name in names]


def _metrics(
    model: LocalAgentLM,
    tokenizer: Any,
    selector: DenseToolSelector,
    tools: list[ToolSpec],
    decisions: list[ProbeDecision],
) -> dict[str, Any]:
    names = [tool.name for tool in tools]
    rows = _tool_rows(decisions, set(names))
    if not rows:
        return {"assistant_decisions": len(decisions), "tool_decisions": 0}
    with torch.no_grad():
        features = torch.stack(
            [_feat(model, tokenizer, row.prompt, "cpu", framed=row.framed) for row in rows]
        )
        scores = selector(features, tool_embeddings(tools, selector.emb_dim))
    result: dict[str, Any] = {
        "assistant_decisions": len(decisions),
        "tool_decisions": len(rows),
    }
    for k in (1, 3, 5, 10):
        hits = sum(
            row.ref_name in [names[index] for index in scores[row_index].topk(min(k, len(names))).indices.tolist()]
            for row_index, row in enumerate(rows)
        )
        result[f"top{k}"] = {"correct": hits, "total": len(rows), "accuracy": hits / len(rows)}
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row_index, row in enumerate(rows):
        if row.ref_name.startswith("browser_"):
            family = "playwright"
        elif row.ref_name.startswith("API-"):
            family = "notion"
        elif row.ref_name in {"list_allowed_directories", "list_directory", "read_text_file", "read_multiple_files", "move_file"}:
            family = "filesystem"
        else:
            family = "other"
        by_family[family]["total"] += 1
        top1 = names[int(scores[row_index].argmax())]
        by_family[family]["correct"] += int(top1 == row.ref_name)
    result["top1_by_family"] = {
        family: {**counts, "accuracy": counts["correct"] / counts["total"]}
        for family, counts in sorted(by_family.items())
    }
    return result


def _relative_state_delta(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name in sorted(left):
        lhs = left[name].detach().float()
        rhs = right[name].detach().float()
        numerator += float((lhs - rhs).pow(2).sum())
        denominator += float(lhs.pow(2).sum())
    return (numerator / denominator) ** 0.5 if denominator else 0.0


def _states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def _run_arm(
    parent: dict[str, Any],
    tokenizer: Any,
    tools: list[ToolSpec],
    train_decisions: list[ProbeDecision],
    eval_decisions: list[ProbeDecision],
    *,
    warm: bool,
    model_seed: int,
    selector_seed: int,
    steps: int,
    batch_size: int,
    proj: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model = _model(parent, warm=warm, seed=model_seed)
    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.manual_seed(selector_seed)
    initial = DenseToolSelector(model.cfg.d_model, emb_dim=8192, proj=proj)
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    torch.manual_seed(selector_seed)
    selector = train_dense_selector(
        model,
        train_decisions,
        tokenizer,
        tools,
        steps=steps,
        batch_size=batch_size,
        proj=proj,
        device="cpu",
        log=lambda *args: None,
    )
    selector_delta = _relative_state_delta(initial_state, selector.state_dict())
    report = {
        "model_state_sha256": state_digest(model_state),
        "selector_state_sha256": state_digest(selector.state_dict()),
        "selector_delta_relative_l2": selector_delta,
        "train": _metrics(model, tokenizer, selector, tools, train_decisions),
        "eval": _metrics(model, tokenizer, selector, tools, eval_decisions),
    }
    return report, model_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proj", type=int, default=256)
    parser.add_argument("--model-seed", type=int, default=2028)
    parser.add_argument("--selector-seed", type=int, default=2027)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.proj < 1:
        raise SystemExit("steps, batch-size, and proj must be positive")
    train_rows = _load_rows(args.train)
    eval_rows = _load_rows(args.eval)
    train_ids = {str(row.meta.get("parent_record_id")) for row in train_rows}
    eval_ids = {str(row.meta.get("parent_record_id")) for row in eval_rows}
    if train_ids & eval_ids:
        raise ValueError(f"train/eval source overlap: {sorted(train_ids & eval_ids)}")
    tools = _tool_catalog(train_rows + eval_rows)
    train_decisions = probe_decisions(train_rows)
    eval_decisions = probe_decisions(eval_rows)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = _checkpoint_tokenizer(parent)
    model_config = dict(parent["cfg"])
    model_config_sha256 = hashlib.sha256(
        json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    parameter_count = sum(int(value.numel()) for value in parent["state_dict"].values())
    warm, warm_state = _run_arm(
        parent,
        tokenizer,
        tools,
        train_decisions,
        eval_decisions,
        warm=True,
        model_seed=args.model_seed,
        selector_seed=args.selector_seed,
        steps=args.steps,
        batch_size=args.batch_size,
        proj=args.proj,
    )
    random, random_state = _run_arm(
        parent,
        tokenizer,
        tools,
        train_decisions,
        eval_decisions,
        warm=False,
        model_seed=args.model_seed,
        selector_seed=args.selector_seed,
        steps=args.steps,
        batch_size=args.batch_size,
        proj=args.proj,
    )
    report = {
        "kind": "localagent_mcpmark_dynamic_selector_transfer",
        "schema_version": 1,
        "evaluator": file_identity(Path(__file__).resolve()),
        "source": {
            "dataset": "Jakumetsu/mcpmark-trajectory-log",
            "url": "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log",
            "revision": "e50578f0ab904d8e6a7c576c387c1e76ae482c89",
            "license": "MIT",
            "train_inputs": [file_identity(path) for path in args.train],
            "eval_inputs": [file_identity(path) for path in args.eval],
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_decisions": len(train_decisions),
            "eval_decisions": len(eval_decisions),
            "tool_catalog": [tool.name for tool in tools],
            "tool_outputs_redacted": True,
            "assistant_free_text_redacted": True,
        },
        "checkpoint": file_identity(args.checkpoint),
        "model_contract": {
            "config": model_config,
            "config_sha256": model_config_sha256,
            "parameter_count": parameter_count,
            "tokenizer": parent.get("tokenizer"),
        },
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "proj": args.proj,
            "model_seed": args.model_seed,
            "selector_seed": args.selector_seed,
            "device": "cpu",
            "frozen_backbone": True,
        },
        "arms": {"warm": warm, "random_backbone": random},
        "weight_analysis": {
            "warm_random_backbone_state_exact": _states_equal(warm_state, random_state),
            "warm_random_backbone_relative_delta_l2": _relative_state_delta(warm_state, random_state),
            "interpretation": (
                "The selector is trained on frozen model features. A warm-vs-random top-k gap is "
                "therefore representation-transfer evidence, while equal or poor top-1 does not "
                "establish native MCP execution."
            ),
        },
        "claim_boundary": (
            "Public redacted MCPMark trajectory routing diagnostic over filesystem/Notion training "
            "rows and a held-out Playwright row. The global catalog contains tool schemas from both "
            "splits; no MCP server, browser, verifier, official split score, or external side effect "
            "was run. This is not an official MCPMark result."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
