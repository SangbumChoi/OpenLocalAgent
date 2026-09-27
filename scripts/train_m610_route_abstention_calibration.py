#!/usr/bin/env python3
"""Calibrate the frozen route head against realistic tools and explicit no-tool prompts.

The m609 browser probe found an over-eager click for a plain definition question.  This experiment
keeps the m585 backbone and dense selector frozen, adds held-out semantic/no-tool probes, and
compares a warm route-head continuation with a matched random route head.  It is a calibration
receipt, not a claim that the entire ONNX bundle has been promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.routes import ROUTES, ROUTE_INDEX, RouteHead, route_of, route_of_sample
from localagent.data.agent_synth import Generator, Sample
from localagent.data.schema import Conversation
from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.stage_data import probe_decisions
from scripts.train_deployment_dispatch_repair import _checkpoint_tokenizer


SEMANTIC_EVAL = (
    ("What does ephemeral mean?", "text"),
    ("Could you explain what a browser cache is?", "text"),
    ("Thanks, that is all for now.", "text"),
    ("Email Dana the quarterly report", "app_action"),
    ("Create a Notion page titled Launch log.", "app_action"),
    ("Search the web for AI news.", "web_search"),
    ("Click the submit button.", "computer_use"),
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _features(model: LocalAgentLM, tokenizer: Any, rows: list[Any]) -> torch.Tensor:
    from localagent.agent.tool_head import _feat

    return torch.stack(
        [
            _feat(
                model,
                tokenizer,
                row.prompt,
                "cpu",
                framed=bool(getattr(row, "framed", False)),
            )
            for row in rows
        ]
    ).detach()


def _explicit_sample(prompt: str, route: str) -> Sample:
    tool = {
        "app_action": "send_email",
        "web_search": "web_search",
        "computer_use": "click",
    }.get(route, "text")
    if tool == "text":
        return Sample("semantic_calibration", "text", prompt, "text", "Acknowledged.")
    return Sample(
        "semantic_calibration",
        route,
        prompt,
        "tool",
        json.dumps({"arguments": {}, "name": tool}, sort_keys=True, separators=(",", ":")),
        tool,
        "{}",
    )


def _build_rows(seed: int, public_paths: list[Path]) -> tuple[list[Any], list[Any], list[dict[str, str]]]:
    train_gen = Generator(level=5, seed=seed, split="train")
    eval_gen = Generator(level=5, seed=seed + 1, split="eval")
    train: list[Any] = []
    evaluation: list[Any] = []
    for _ in range(96):
        train.extend((train_gen.no_tool(), train_gen.text()))
    for _ in range(48):
        evaluation.extend((eval_gen.no_tool(), eval_gen.text()))
    # Use paraphrased deployment prompts in calibration and reserve the exact canonical strings
    # below for evaluation.  The tool rows are important: text/no-tool calibration must not
    # collapse the five-way head into the text class.
    train.extend(
        _explicit_sample(prompt, route)
        for prompt, route in (
            ("Explain ephemeral in one sentence.", "text"),
            ("Please do not perform any action; just acknowledge this note.", "text"),
            ("Send the quarterly report to the release owner.", "app_action"),
            ("Create a Notion page titled Weekly log.", "app_action"),
            ("Look up the latest AI news on the web.", "web_search"),
            ("Click the Login button.", "computer_use"),
            ("Click the Save icon.", "computer_use"),
            ("Tap the submit button.", "computer_use"),
            ("Use the mouse to click Search.", "computer_use"),
            ("Type hello into the form.", "computer_use"),
        )
    )
    evaluation.extend(_explicit_sample(prompt, route) for prompt, route in SEMANTIC_EVAL)

    public_identities: list[dict[str, str]] = []
    for path in public_paths:
        conversations: list[Conversation] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    conversations.append(Conversation.from_json(line))
        decisions = probe_decisions(conversations)
        # Public projections provide realistic tool-route diversity; cap each file so the
        # no-tool calibration cannot be drowned out by thousands of tool rows.
        train.extend(decisions[:128])
        public_identities.append(_identity(path) | {"rows": len(conversations)})
    return train, evaluation, public_identities


def _label(row: Any) -> int:
    if isinstance(row, Sample):
        return ROUTE_INDEX[route_of_sample(row)]
    return ROUTE_INDEX[route_of(str(getattr(row, "ref_name", "text")))]


def _train_arm(
    model: LocalAgentLM,
    tokenizer: Any,
    parent: dict[str, Any],
    rows: list[Any],
    *,
    warm: bool,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[RouteHead, torch.Tensor]:
    torch.manual_seed(seed)
    route = RouteHead(model.cfg.d_model)
    if warm:
        route.load_state_dict(parent["route_head"])
    features = _features(model, tokenizer, rows)
    labels = torch.tensor([_label(row) for row in rows], dtype=torch.long)
    teacher = RouteHead(model.cfg.d_model)
    teacher.load_state_dict(parent["route_head"])
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(features)
    # Inverse-frequency weights keep the explicit text/no-tool class visible despite public
    # tool-heavy projections.
    counts = torch.bincount(labels, minlength=len(ROUTES)).float().clamp_min(1.0)
    weights = (labels.numel() / counts / len(ROUTES)).to(dtype=features.dtype)
    optimizer = torch.optim.AdamW(route.parameters(), lr=lr)
    rng = random.Random(seed)
    class_rows = {
        class_index: [index for index, label in enumerate(labels.tolist()) if label == class_index]
        for class_index in sorted(set(labels.tolist()))
    }
    active_classes = tuple(class_rows)
    route.train()
    for _ in range(steps):
        per_class = max(1, batch_size // max(1, len(active_classes)))
        indices = [
            rng.choice(class_rows[class_index])
            for class_index in active_classes
            for _ in range(per_class)
        ]
        index = torch.tensor(indices, dtype=torch.long)
        logits = route(features[index])
        loss = F.cross_entropy(logits, labels[index], weight=weights)
        # Preserve the existing GUI boundary while moving the text/no-tool boundary.  The
        # browser's explicit email/Notion guards cover side-effecting names, but a generic click
        # must remain computer_use after calibration.
        computer = labels[index] == ROUTE_INDEX["computer_use"]
        if bool(computer.any()):
            loss = loss + 0.75 * F.mse_loss(
                logits[computer], teacher_logits[index][computer]
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    route.eval()
    return route, features


def _score(route: RouteHead, model: LocalAgentLM, tokenizer: Any, rows: list[Any]) -> dict[str, Any]:
    features = _features(model, tokenizer, rows)
    labels = [_label(row) for row in rows]
    predictions = route(features).argmax(-1).tolist()
    return {
        "rows": len(rows),
        "accuracy": sum(int(pred == label) for pred, label in zip(predictions, labels)) / max(1, len(rows)),
        "by_route": {
            route_name: {
                "rows": sum(label == index for label in labels),
                "accuracy": sum(pred == label for pred, label in zip(predictions, labels) if label == index)
                / max(1, sum(label == index for label in labels)),
            }
            for index, route_name in enumerate(ROUTES)
        },
    }


def _semantic_score(route: RouteHead, model: LocalAgentLM, tokenizer: Any) -> list[dict[str, Any]]:
    rows = [SimpleNamespace(prompt=prompt, ref_name={
        "text": "text",
        "app_action": "send_email",
        "web_search": "web_search",
        "computer_use": "click",
    }[expected], framed=False) for prompt, expected in SEMANTIC_EVAL]
    features = _features(model, tokenizer, rows)
    predictions = route(features).argmax(-1).tolist()
    return [
        {
            "prompt": prompt,
            "expected_route": expected,
            "predicted_route": ROUTES[prediction],
            "route_exact": ROUTES[prediction] == expected,
        }
        for (prompt, expected), prediction in zip(SEMANTIC_EVAL, predictions)
    ]


def _movement(parent: dict[str, Any], child: RouteHead) -> float:
    before = parent["route_head"]
    after = child.state_dict()
    numerator = sum(float((after[name].float() - value.float()).pow(2).sum()) for name, value in before.items())
    denominator = sum(float(value.float().pow(2).sum()) for value in before.values())
    return (numerator**0.5) / max(denominator**0.5, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--public-train", type=Path, action="append", default=[])
    parser.add_argument("--warm-output", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=2061)
    args = parser.parse_args()
    if any(path.exists() for path in (args.warm_output, args.random_output, args.report)):
        raise SystemExit("refusing to overwrite output")
    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    config = ModelConfig(**parent["cfg"])
    config.assert_within_budget()
    model = LocalAgentLM(config).eval()
    model.load_state_dict(parent["state_dict"])
    tokenizer = _checkpoint_tokenizer(parent)
    train_rows, eval_rows, public_identities = _build_rows(args.seed, args.public_train)
    warm, _ = _train_arm(model, tokenizer, parent, train_rows, warm=True, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    random_arm, _ = _train_arm(model, tokenizer, parent, train_rows, warm=False, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    parent_identity = _identity(args.init)
    warm_child = dict(parent)
    warm_child.update({"route_head": warm.state_dict(), "stage": "sft_m610_route_abstention_calibration", "parent_checkpoint_sha256": parent_identity["sha256"]})
    random_child = dict(parent)
    random_child.update({"route_head": random_arm.state_dict(), "stage": "sft_m610_route_abstention_calibration", "parent_checkpoint_sha256": parent_identity["sha256"]})
    args.warm_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warm_child, args.warm_output)
    torch.save(random_child, args.random_output)
    warm_eval = _score(warm, model, tokenizer, eval_rows)
    random_eval = _score(random_arm, model, tokenizer, eval_rows)
    report: dict[str, Any] = {
        "kind": "localagent_m610_route_abstention_calibration",
        "schema_version": 1,
        "parent": parent_identity,
        "warm_child": _identity(args.warm_output),
        "random_child": _identity(args.random_output),
        "data": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "public_train": public_identities,
            "semantic_eval": list(SEMANTIC_EVAL),
            "text_and_no_tool_rows_in_train": sum(_label(row) == ROUTE_INDEX["text"] for row in train_rows),
        },
        "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed, "backbone_frozen": True, "selector_frozen": True},
        "evaluation": {"warm": warm_eval, "random": random_eval, "warm_semantic": _semantic_score(warm, model, tokenizer), "random_semantic": _semantic_score(random_arm, model, tokenizer)},
        "weight_movement": {"warm_route_head_relative_l2": _movement(parent, warm), "random_route_head_relative_l2": _movement(parent, random_arm), "backbone_relative_l2": 0.0, "dense_selector_relative_l2": 0.0},
        "decision": {"warm_semantic_exact": all(row["route_exact"] for row in _semantic_score(warm, model, tokenizer)), "random_semantic_exact": all(row["route_exact"] for row in _semantic_score(random_arm, model, tokenizer)), "adopt_warm_route_head": False, "export_to_webgpu": False},
        "claim_boundary": "Frozen route-head calibration on public text/accessibility projections plus explicit no-tool/semantic adapter rows. No screenshots, emulator/VM, browser account, email/Notion side effect, official benchmark score, or public export is claimed.",
    }
    report["receipt_self_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
