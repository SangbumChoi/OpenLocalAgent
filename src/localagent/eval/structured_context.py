"""Context-length robustness audit for frozen structured-action probes.

The browser complete-action harness can materialize a prompt at an exact tokenizer length by
inserting single-token spaces immediately before the assistant marker.  That is a useful systems
stress condition, but route and selector probes trained only on short natural prompts may not
transfer to it.  This module evaluates the frozen SFT heads under both conditions without changing
the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector
from localagent.agent.routes import ROUTES, RouteHead, route_of
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.stage_data import ProbeDecision, probe_decisions, read_conversations


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return an exact byte identity for a local input artifact."""

    resolved = Path(path)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def canonical_sha256(value: Any) -> str:
    """Hash canonical compact JSON."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def materialize_context_ids(
    tokenizer: Any,
    text: str,
    *,
    target_input_tokens: int | None,
    max_seq_len: int,
) -> list[int] | None:
    """Encode one framed prompt and optionally reproduce browser fixed-length padding.

    ``None`` means the natural prompt, left-truncated to the model context window.  A fixed target
    mirrors ``spaces/localagent-webgpu/app.js::padPromptIds``: prompts longer than the target are
    ineligible; otherwise one-token spaces are inserted immediately before the final assistant
    marker.  The assistant suffix is rechecked rather than assumed.
    """

    materialized = materialize_context_view(
        tokenizer,
        text,
        target_input_tokens=target_input_tokens,
        max_seq_len=max_seq_len,
        materialization="pre_assistant",
    )
    return None if materialized is None else materialized[0]


def materialize_context_view(
    tokenizer: Any,
    text: str,
    *,
    target_input_tokens: int | None,
    max_seq_len: int,
    materialization: str,
) -> tuple[list[int], int] | None:
    """Return model IDs and the hidden-state index consumed by the structured heads.

    ``pre_assistant`` reproduces the original browser stress arm: filler precedes the assistant
    marker and the heads consume the final state. ``trailing_compute`` appends filler after the
    complete natural prompt but consumes the natural assistant-marker state. Causality makes the
    latter a fixed-compute latency condition without changing the policy feature view.
    """

    if materialization not in {"pre_assistant", "trailing_compute"}:
        raise ValueError(f"unknown context materialization: {materialization}")
    ids = tokenizer.encode(text)
    if target_input_tokens is None:
        natural = ids[-max_seq_len:]
        return natural, len(natural) - 1
    if target_input_tokens < 1 or target_input_tokens > max_seq_len:
        raise ValueError("target_input_tokens must be within the model context window")
    if len(ids) > target_input_tokens:
        return None
    assistant_ids = tokenizer.encode(ASSISTANT)
    if not assistant_ids or ids[-len(assistant_ids) :] != assistant_ids:
        raise ValueError("framed prompt does not end with the assistant marker")
    whitespace_ids = tokenizer.encode(" ")
    if len(whitespace_ids) != 1:
        raise ValueError("tokenizer must encode one space as exactly one token")
    padding = [whitespace_ids[0]] * (target_input_tokens - len(ids))
    if materialization == "pre_assistant":
        result = [*ids[: -len(assistant_ids)], *padding, *assistant_ids]
        feature_index = len(result) - 1
    else:
        result = [*ids, *padding]
        feature_index = len(ids) - 1
    if len(result) != target_input_tokens:
        raise AssertionError("fixed-context materialization missed its exact token target")
    return result, feature_index


def _feature_matrix(
    model: LocalAgentLM,
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    batch_size: int,
    device: str,
    feature_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    if not sequences:
        return torch.empty((0, model.cfg.d_model), device=device)
    rows: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            lengths = torch.tensor([len(ids) for ids in batch], device=device)
            width = int(lengths.max().item())
            inputs = torch.full(
                (len(batch), width),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            for row, ids in enumerate(batch):
                inputs[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            _, hidden = model(inputs, return_hidden=True)
            row_index = torch.arange(len(batch), device=device)
            if feature_indices is None:
                selected = lengths - 1
            else:
                selected = torch.tensor(
                    feature_indices[start : start + len(batch)],
                    dtype=torch.long,
                    device=device,
                )
            rows.append(hidden[row_index, selected])
    return torch.cat(rows, dim=0)


def _condition_label(target_input_tokens: int | None, materialization: str) -> str:
    if target_input_tokens is None:
        return "natural"
    return f"fixed_{materialization}_{target_input_tokens}"


def _score_rows(
    *,
    features: torch.Tensor,
    gold_routes: Sequence[str],
    gold_tools: Sequence[str | None],
    route_head: torch.nn.Module,
    selector: BoundSelector,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(features) != len(gold_routes) or len(features) != len(gold_tools):
        raise ValueError("feature and label row counts differ")
    with torch.no_grad():
        route_logits = route_head(features)
        route_probabilities = torch.softmax(route_logits, dim=-1)
        route_indices = route_logits.argmax(dim=-1)
        selector_logits = selector.model(features, selector.embs)
        selector_indices = selector_logits.argmax(dim=-1)

    records: list[dict[str, Any]] = []
    route_correct = 0
    selector_correct = 0
    dispatched_correct = 0
    text_predictions = 0
    tool_rows = 0
    for index, (gold_route, gold_tool) in enumerate(zip(gold_routes, gold_tools, strict=True)):
        route_index = int(route_indices[index].item())
        predicted_route = ROUTES[route_index]
        predicted_tool = selector.names[int(selector_indices[index].item())]
        is_tool = gold_tool is not None
        route_ok = predicted_route == gold_route
        selector_ok = bool(is_tool and predicted_tool == gold_tool)
        dispatched_ok = bool(selector_ok and predicted_route != "text")
        route_correct += route_ok
        selector_correct += selector_ok
        dispatched_correct += dispatched_ok
        text_predictions += predicted_route == "text"
        tool_rows += is_tool
        records.append(
            {
                "gold_route": gold_route,
                "gold_tool": gold_tool,
                "predicted_route": predicted_route,
                "predicted_route_confidence": float(
                    route_probabilities[index, route_index].item()
                ),
                "predicted_tool": predicted_tool,
                "route_correct": route_ok,
                "selector_top1_correct": selector_ok,
                "dispatched_tool_correct": dispatched_ok,
            }
        )
    total = len(records)
    metrics = {
        "eligible_rows": total,
        "route_correct": route_correct,
        "route_accuracy": route_correct / max(1, total),
        "route_text_predictions": text_predictions,
        "tool_rows": tool_rows,
        "selector_top1_correct": selector_correct,
        "selector_top1_accuracy": selector_correct / max(1, tool_rows),
        "dispatched_tool_correct": dispatched_correct,
        "dispatched_tool_accuracy": dispatched_correct / max(1, tool_rows),
    }
    return metrics, records


def _framed_decision_text(decision: ProbeDecision) -> str:
    return decision.prompt if decision.framed else f"{USER}{decision.prompt}{ASSISTANT}"


def evaluate_decisions(
    *,
    model: LocalAgentLM,
    tokenizer: Any,
    route_head: torch.nn.Module,
    selector: BoundSelector,
    decisions: Sequence[ProbeDecision],
    target_input_tokens: int | None,
    batch_size: int,
    device: str,
    materialization: str = "pre_assistant",
    include_records: bool = False,
) -> dict[str, Any]:
    """Evaluate canonical assistant decisions at one context condition."""

    sequences: list[list[int]] = []
    feature_indices: list[int] = []
    eligible_indices: list[int] = []
    eligible: list[ProbeDecision] = []
    for configured_index, decision in enumerate(decisions):
        view = materialize_context_view(
            tokenizer,
            _framed_decision_text(decision),
            target_input_tokens=target_input_tokens,
            max_seq_len=model.cfg.max_seq_len,
            materialization=materialization,
        )
        if view is None:
            continue
        ids, feature_index = view
        sequences.append(ids)
        feature_indices.append(feature_index)
        eligible_indices.append(configured_index)
        eligible.append(decision)
    features = _feature_matrix(
        model,
        sequences,
        pad_id=tokenizer.pad_id,
        batch_size=batch_size,
        device=device,
        feature_indices=feature_indices,
    )
    metrics, records = _score_rows(
        features=features,
        gold_routes=[
            "text" if decision.kind != "tool" else route_of(decision.ref_name)
            for decision in eligible
        ],
        gold_tools=[
            decision.ref_name if decision.kind == "tool" else None for decision in eligible
        ],
        route_head=route_head,
        selector=selector,
    )
    metrics.update(
        {
            "condition": _condition_label(target_input_tokens, materialization),
            "materialization": (
                "natural" if target_input_tokens is None else materialization
            ),
            "configured_rows": len(decisions),
            "ineligible_rows": len(decisions) - len(eligible),
        }
    )
    if include_records:
        metrics["records"] = [
            {"configured_index": configured_index, **record}
            for configured_index, record in zip(eligible_indices, records, strict=True)
        ]
    return metrics


def evaluate_action_cases(
    *,
    model: LocalAgentLM,
    tokenizer: Any,
    route_head: torch.nn.Module,
    selector: BoundSelector,
    cases: Sequence[dict[str, Any]],
    target_input_tokens: int | None,
    batch_size: int,
    device: str,
    materialization: str = "pre_assistant",
) -> dict[str, Any]:
    """Evaluate the frozen browser action suite through route and tool-selection stages."""

    sequences: list[list[int]] = []
    feature_indices: list[int] = []
    eligible: list[dict[str, Any]] = []
    for case in cases:
        view = materialize_context_view(
            tokenizer,
            f"{USER}{case['query']}{ASSISTANT}",
            target_input_tokens=target_input_tokens,
            max_seq_len=model.cfg.max_seq_len,
            materialization=materialization,
        )
        if view is None:
            continue
        ids, feature_index = view
        sequences.append(ids)
        feature_indices.append(feature_index)
        eligible.append(case)
    features = _feature_matrix(
        model,
        sequences,
        pad_id=tokenizer.pad_id,
        batch_size=batch_size,
        device=device,
        feature_indices=feature_indices,
    )
    gold_tools = [case["expected"].get("tool") for case in eligible]
    metrics, scored = _score_rows(
        features=features,
        gold_routes=["text" if tool is None else route_of(tool) for tool in gold_tools],
        gold_tools=gold_tools,
        route_head=route_head,
        selector=selector,
    )
    records = []
    for case, score in zip(eligible, scored, strict=True):
        records.append(
            {
                "case_id": case["id"],
                "family": case["family"],
                **score,
            }
        )
    metrics.update(
        {
            "condition": _condition_label(target_input_tokens, materialization),
            "materialization": (
                "natural" if target_input_tokens is None else materialization
            ),
            "configured_rows": len(cases),
            "ineligible_rows": len(cases) - len(eligible),
            "records": records,
        }
    )
    return metrics


def _load_action_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("action suite must be an object containing a cases list")
    return payload["cases"]


def build_context_audit(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    agent_eval_path: str | Path,
    action_suite_path: str | Path,
    fixed_contexts: Iterable[int] = (128, 512),
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load one SFT checkpoint and build a reproducible natural-vs-fixed context audit."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("route_head") is None or checkpoint.get("dense_selector") is None:
        raise ValueError("checkpoint lacks route_head or dense_selector")
    cfg = ModelConfig(**checkpoint["cfg"])
    cfg.assert_within_budget()
    tokenizer = load_tokenizer("bpe", str(tokenizer_path))
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint config")

    model = LocalAgentLM(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    route_head = RouteHead(cfg.d_model).to(device)
    route_head.load_state_dict(checkpoint["route_head"])
    route_head.eval()
    selector_model = DenseToolSelector(
        cfg.d_model,
        proj=int(checkpoint["selector_proj"]),
    ).to(device)
    selector_model.load_state_dict(checkpoint["dense_selector"])
    selector = BoundSelector(
        selector_model,
        STANDARD_TOOLS,
        device=device,
        examples=checkpoint.get("examples"),
    )

    decisions = probe_decisions(read_conversations(agent_eval_path))
    cases = _load_action_cases(action_suite_path)
    condition_specs: list[tuple[int | None, str]] = [(None, "pre_assistant")]
    for value in fixed_contexts:
        target = int(value)
        condition_specs.extend(
            [
                (target, "pre_assistant"),
                (target, "trailing_compute"),
            ]
        )
    conditions = []
    for target, materialization in condition_specs:
        conditions.append(
            {
                "condition": _condition_label(target, materialization),
                "target_input_tokens": target,
                "materialization": (
                    "natural" if target is None else materialization
                ),
                "agent_eval": evaluate_decisions(
                    model=model,
                    tokenizer=tokenizer,
                    route_head=route_head,
                    selector=selector,
                    decisions=decisions,
                    target_input_tokens=target,
                    batch_size=batch_size,
                    device=device,
                    materialization=materialization,
                ),
                "action_suite": evaluate_action_cases(
                    model=model,
                    tokenizer=tokenizer,
                    route_head=route_head,
                    selector=selector,
                    cases=cases,
                    target_input_tokens=target,
                    batch_size=batch_size,
                    device=device,
                    materialization=materialization,
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "localagent_structured_context_robustness_audit",
        "execution": {
            "device": device,
            "dtype": "fp32",
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "checkpoint": {
            **file_identity(checkpoint_path),
            "stage": checkpoint.get("stage"),
            "step": checkpoint.get("step"),
            "model_name": cfg.name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "inputs": {
            "tokenizer": file_identity(tokenizer_path),
            "agent_eval": file_identity(agent_eval_path),
            "action_suite": file_identity(action_suite_path),
        },
        "contract": {
            "natural": "canonical tokenizer IDs with model-window left truncation only",
            "fixed": (
                "single-token spaces inserted immediately before the final assistant marker, "
                "matching the original browser complete-action stress condition"
            ),
            "trailing_compute": (
                "single-token spaces appended after the natural assistant marker while the "
                "structured heads consume that marker state; this fixes compute length without "
                "changing the natural policy feature view"
            ),
            "scope": (
                "frozen PyTorch route and dense-selector diagnostic; no pointer arguments, "
                "schema validation, DOM execution, WebGPU timing, or retraining"
            ),
        },
        "conditions": conditions,
        "summary_sha256": "",
    }
    payload["summary_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )
    return payload


def write_context_audit(payload: dict[str, Any], path: str | Path) -> None:
    """Write an audit as deterministic pretty JSON."""

    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
