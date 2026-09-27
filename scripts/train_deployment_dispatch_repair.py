#!/usr/bin/env python3
"""Repair deployment dispatch on public cross-surface rows plus productivity prompts.

The warm m103 head probe was trained on AndroidControl, AgentNet, and Mind2Web projections, so
it had no email/Notion app-action coverage.  This runner keeps the language-model backbone fixed,
adds deterministic productivity/browser paraphrases, and trains matched warm-start and random-head
arms on the same cached features.  The public rows remain source-disjoint train/eval evidence;
the generated productivity rows are a deployment adapter and are never reported as a public
benchmark score or real-account execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.routes import ROUTES, ROUTE_INDEX, RouteHead, route_of
from localagent.agent.toolset import REALISTIC_BROWSER_TOOLS
from localagent.data.agent_synth import Generator, Sample
from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.stage_data import probe_decisions


CORE_REPAIR_TOOLS = (
    "send_email",
    "notion_write",
    "web_search",
    "open_url",
    "calendar_event",
    "slack_send",
    "click",
    "type_text",
)

CANONICAL_PROBES = (
    ("Email Dana the quarterly report", "send_email", "app_action"),
    (
        "Send an email to alice@example.com with subject WebGPU test and body Local bundle verified.",
        "send_email",
        "app_action",
    ),
    (
        "Create a Notion page titled Launch log with content Verify the browser gate.",
        "notion_write",
        "app_action",
    ),
    ("Search the web for AI news, then save it to Notion.", "web_search", "web_search"),
    ("Open https://github.com/pytorch/pytorch", "open_url", "web_search"),
    ("Click the submit button", "click", "computer_use"),
)


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _sample_identity(samples: list[Any]) -> str:
    rows = [
        {
            "kind": str(getattr(sample, "kind", "")),
            "prompt": str(getattr(sample, "prompt", "")),
            "ref_name": str(getattr(sample, "ref_name", "")),
        }
        for sample in samples
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_tokenizer(checkpoint: dict[str, Any]):
    metadata = checkpoint.get("tokenizer") or {"kind": "byte"}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")
    kind = str(metadata.get("kind", "byte"))
    path = metadata.get("path")
    if kind == "bpe" and path is None:
        raise ValueError("BPE checkpoint is missing tokenizer.path")
    tokenizer = load_tokenizer(kind, path)
    cfg = checkpoint["cfg"]
    vocab_size = cfg["vocab_size"] if isinstance(cfg, dict) else cfg.vocab_size
    if tokenizer.vocab_size != vocab_size:
        raise ValueError("checkpoint tokenizer vocabulary does not match model config")
    return tokenizer


def _load_public_decisions(paths: list[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    decisions: list[Any] = []
    identities: list[dict[str, Any]] = []
    for path in paths:
        conversations = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    conversations.append(Conversation.from_json(line))
                except Exception as error:  # pragma: no cover - diagnostic context
                    raise ValueError(f"invalid Conversation at {path}:{line_number}") from error
        if not conversations:
            raise ValueError(f"public input is empty: {path}")
        decisions.extend(probe_decisions(conversations))
        identities.append(_sha256(path) | {"rows": len(conversations)})
    return decisions, identities


def _repair_sample(tool: str, prompt: str, value: str = "deployment adapter") -> Sample:
    argument = {
        "send_email": "recipient",
        "notion_write": "content",
        "web_search": "query",
        "open_url": "url",
        "calendar_event": "title",
        "slack_send": "message",
        "click": "target",
        "type_text": "text",
    }.get(tool)
    args = {} if argument is None else {argument: value}
    return Sample(
        "deployment_dispatch_repair",
        "repair",
        prompt,
        "tool",
        json.dumps({"arguments": args, "name": tool}, sort_keys=True, separators=(",", ":")),
        tool,
        json.dumps(args, sort_keys=True, separators=(",", ":")),
    )


def _synthetic_rows(seed: int, per_core: int) -> tuple[list[Any], list[Any], dict[str, list[str]]]:
    train_gen = Generator(level=5, seed=seed, split="train")
    eval_gen = Generator(level=5, seed=seed + 1, split="eval")
    method_by_tool = {
        "send_email": "send_email",
        "notion_write": "notion_write",
        "web_search": "web_search",
        "open_url": "open_url",
        "calendar_event": "calendar_event",
        "slack_send": "slack_send",
        "click": "click",
        "type_text": "type_text",
    }
    train: list[Any] = []
    evaluation: list[Any] = []
    examples: dict[str, list[str]] = defaultdict(list)
    for tool in CORE_REPAIR_TOOLS:
        maker = method_by_tool[tool]
        for _ in range(per_core):
            sample = getattr(train_gen, maker)()
            train.append(sample)
            examples[tool].append(sample.prompt)
        for _ in range(max(4, per_core // 4)):
            evaluation.append(getattr(eval_gen, maker)())

    # Explicit deployment-shaped paraphrases are train-only adapter rows; their slot values are
    # intentionally different from the browser smoke prompts below.
    train.extend(
        [
            _repair_sample("send_email", "Email the launch owner about the release.", "the launch owner"),
            _repair_sample("send_email", "Send the project update to the release manager.", "the release manager"),
            _repair_sample("notion_write", "Create a Notion page with the launch notes.", "the launch notes"),
            _repair_sample("notion_write", "Save the sprint summary to Notion.", "the sprint summary"),
            _repair_sample("web_search", "Search the web for the latest AI news.", "the latest AI news"),
            _repair_sample("open_url", "Open the release dashboard in the browser.", "the release dashboard"),
        ]
    )
    examples["send_email"].extend(
        ["Email the launch owner about the release.", "Send the project update to the release manager."]
    )
    examples["notion_write"].extend(
        ["Create a Notion page with the launch notes.", "Save the sprint summary to Notion."]
    )
    examples["web_search"].append("Search the web for the latest AI news.")
    examples["open_url"].append("Open the release dashboard in the browser.")
    return train, evaluation, dict(examples)


def _features(model: LocalAgentLM, tokenizer: Any, decisions: list[Any], batch_size: int = 32) -> torch.Tensor:
    """Cache final prompt features in batches, preserving the single-row ``_feat`` contract."""

    encoded = [
        tokenizer.encode(
            decision.prompt
            if bool(getattr(decision, "framed", False))
            else f"{USER}{decision.prompt}{ASSISTANT}"
        )[-model.cfg.max_seq_len :]
        for decision in decisions
    ]
    features: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            rows = encoded[start : start + batch_size]
            width = max(len(row) for row in rows)
            ids = torch.zeros((len(rows), width), dtype=torch.long)
            for index, row in enumerate(rows):
                ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            _logits, hidden = model(ids, return_hidden=True)
            features.extend(hidden[index, len(row) - 1].detach() for index, row in enumerate(rows))
    return torch.stack(features)


def _train_arm(
    model: LocalAgentLM,
    tokenizer: Any,
    parent: dict[str, Any],
    decisions: list[Any],
    examples: dict[str, list[str]],
    *,
    warm: bool,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[RouteHead, DenseToolSelector]:
    torch.manual_seed(seed)
    route = RouteHead(model.cfg.d_model)
    selector_state = parent["dense_selector"]
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=selector_state["t_proj.weight"].shape[1],
        proj=int(parent.get("selector_proj", selector_state["q_proj.weight"].shape[0])),
    )
    if warm:
        route.load_state_dict(parent["route_head"])
        selector.load_state_dict(selector_state)
    model.eval()
    route.eval()
    selector.eval()
    features = _features(model, tokenizer, decisions)
    route_labels = torch.tensor(
        [ROUTE_INDEX[route_of(decision.ref_name)] for decision in decisions], dtype=torch.long
    )
    tool_index = {tool.name: index for index, tool in enumerate(REALISTIC_BROWSER_TOOLS)}
    selector_rows = [
        index
        for index, decision in enumerate(decisions)
        if decision.kind == "tool" and decision.ref_name in tool_index
    ]
    if not selector_rows:
        raise ValueError("dispatch repair has no tool decisions")
    selector_features = features[selector_rows]
    selector_labels = torch.tensor([tool_index[decisions[index].ref_name] for index in selector_rows])
    embeddings = tool_embeddings(
        REALISTIC_BROWSER_TOOLS,
        device="cpu",
        examples=examples,
    )
    optimizer = torch.optim.AdamW([*route.parameters(), *selector.parameters()], lr=lr)
    rng = random.Random(seed)
    route.train()
    selector.train()
    for _ in range(steps):
        route_indices = [rng.randrange(len(decisions)) for _ in range(min(batch_size, len(decisions)))]
        selector_indices = [rng.randrange(len(selector_rows)) for _ in range(min(batch_size, len(selector_rows)))]
        loss = F.cross_entropy(route(features[route_indices]), route_labels[route_indices])
        loss = loss + F.cross_entropy(
            selector(selector_features[selector_indices], embeddings), selector_labels[selector_indices]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    route.eval()
    selector.eval()
    return route, selector


def _score(
    model: LocalAgentLM,
    tokenizer: Any,
    route: RouteHead,
    selector: DenseToolSelector,
    examples: dict[str, list[str]],
    decisions: list[Any],
) -> dict[str, Any]:
    if not decisions:
        return {"rows": 0, "route_accuracy": 0.0, "selector_top1_accuracy": 0.0, "tool_rows": 0}
    features = _features(model, tokenizer, decisions)
    embeddings = tool_embeddings(REALISTIC_BROWSER_TOOLS, examples=examples)
    names = [tool.name for tool in REALISTIC_BROWSER_TOOLS]
    with torch.no_grad():
        route_predictions = route(features).argmax(-1).tolist()
        selector_predictions = selector(features, embeddings).argmax(-1).tolist()
    route_correct = sum(ROUTES[pred] == route_of(decision.ref_name) for pred, decision in zip(route_predictions, decisions))
    tool_rows = [index for index, decision in enumerate(decisions) if decision.kind == "tool" and decision.ref_name in names]
    selector_correct = sum(names[selector_predictions[index]] == decisions[index].ref_name for index in tool_rows)
    return {
        "rows": len(decisions),
        "route_accuracy": route_correct / len(decisions),
        "selector_top1_accuracy": selector_correct / len(tool_rows) if tool_rows else 0.0,
        "tool_rows": len(tool_rows),
    }


def _canonical_score(model, tokenizer, route, selector, examples):
    bound = BoundSelector(selector, REALISTIC_BROWSER_TOOLS, examples=examples)
    results = []
    for prompt, expected_tool, expected_route in CANONICAL_PROBES:
        feature = _features(
            model,
            tokenizer,
            [SimpleNamespace(prompt=prompt, framed=False)],
            batch_size=1,
        )[0]
        with torch.no_grad():
            route_name = ROUTES[route(feature.unsqueeze(0)).argmax(-1).item()]
        ranked = bound.rank(feature)
        results.append(
            {
                "prompt": prompt,
                "expected_route": expected_route,
                "expected_tool": expected_tool,
                "predicted_route": route_name,
                "predicted_tool": ranked[0],
                "top3": ranked[:3],
                "route_exact": route_name == expected_route,
                "tool_exact": ranked[0] == expected_tool,
            }
        )
    return results


def _head_movement(parent: dict[str, Any], route: RouteHead, selector: DenseToolSelector) -> dict[str, float]:
    def relative(before, after):
        numerator = 0.0
        denominator = 0.0
        for name, value in before.items():
            left = value.detach().float()
            right = after[name].detach().float()
            numerator += float((right - left).pow(2).sum())
            denominator += float(left.pow(2).sum())
        return (numerator**0.5) / max(denominator**0.5, 1e-12)

    parent_route = parent["route_head"]
    parent_selector = parent["dense_selector"]
    return {
        "route_head_relative_l2": relative(parent_route, route.state_dict()),
        "dense_selector_relative_l2": relative(parent_selector, selector.state_dict()),
        "backbone_relative_l2": 0.0,
    }


def _identity(path: Path) -> dict[str, Any]:
    return _sha256(path)


def _write_arm(parent: dict[str, Any], parent_identity: dict[str, Any], path: Path, route, selector, examples, arm: str):
    child = dict(parent)
    child.update(
        {
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "examples": examples,
            "retrieval_examples": examples,
            "stage": "sft_deployment_dispatch_repair",
            "parent_checkpoint_sha256": parent_identity["sha256"],
            "deployment_dispatch_repair": {"arm": arm, "backbone_frozen": True},
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--public-train", type=Path, action="append", default=[])
    parser.add_argument("--public-eval", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--synthetic-per-core", type=int, default=160)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.synthetic_per_core < 4:
        raise SystemExit("steps, batch-size, lr, and synthetic-per-core are invalid")
    for path in (args.output, args.random_output, args.report):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config).eval()
    model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)
    parent_identity = _identity(args.init)
    public_train, train_identities = _load_public_decisions(args.public_train)
    public_eval, eval_identities = _load_public_decisions(args.public_eval)
    synthetic_train, synthetic_eval, repair_examples = _synthetic_rows(args.seed, args.synthetic_per_core)
    train_decisions = [*public_train, *synthetic_train]
    eval_decisions = [*public_eval, *synthetic_eval]
    examples = dict(parent.get("examples", {}))
    for tool, prompts in repair_examples.items():
        examples.setdefault(tool, []).extend(prompts)
    train_identity = _sample_identity(train_decisions)
    eval_identity = _sample_identity(eval_decisions)

    warm_route, warm_selector = _train_arm(
        model, tokenizer, parent, train_decisions, examples, warm=True,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    random_route, random_selector = _train_arm(
        model, tokenizer, parent, train_decisions, examples, warm=False,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    warm_eval = _score(model, tokenizer, warm_route, warm_selector, examples, eval_decisions)
    random_eval = _score(model, tokenizer, random_route, random_selector, examples, eval_decisions)
    warm_canonical = _canonical_score(model, tokenizer, warm_route, warm_selector, examples)
    random_canonical = _canonical_score(model, tokenizer, random_route, random_selector, examples)
    _write_arm(parent, parent_identity, args.output, warm_route, warm_selector, examples, "warm")
    _write_arm(parent, parent_identity, args.random_output, random_route, random_selector, examples, "random")
    report = {
        "kind": "localagent_deployment_dispatch_repair",
        "schema_version": 1,
        "parent": parent_identity,
        "warm_child": _identity(args.output),
        "random_child": _identity(args.random_output),
        "data": {
            "public_train": train_identities,
            "public_eval": eval_identities,
            "public_train_rows": len(public_train),
            "public_eval_rows": len(public_eval),
            "synthetic_train_rows": len(synthetic_train),
            "synthetic_eval_rows": len(synthetic_eval),
            "train_decision_sha256": train_identity,
            "eval_decision_sha256": eval_identity,
            "repair_tools": list(CORE_REPAIR_TOOLS),
            "synthetic_adapter_only": True,
        },
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "backbone_frozen": True,
        },
        "evaluation": {
            "warm_public_plus_synthetic": warm_eval,
            "random_public_plus_synthetic": random_eval,
            "warm_canonical": warm_canonical,
            "random_canonical": random_canonical,
        },
        "weight_movement": {
            "warm": _head_movement(parent, warm_route, warm_selector),
            "random": _head_movement(parent, random_route, random_selector),
        },
        "claim_boundary": (
            "Frozen dispatch-head repair with public text/accessibility rows plus a deterministic local "
            "productivity adapter. It does not establish official benchmark success, visual grounding, "
            "native Android/desktop/MCP execution, real email/Notion side effects, or public Hub upload."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
