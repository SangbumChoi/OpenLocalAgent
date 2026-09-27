"""Leakage-safe tool-retrieval evaluation for public EnterpriseOps-Gym rows.

EnterpriseOps-Gym is an execution benchmark, not a LocalAgent training corpus.  This adapter
consumes only the public task/tool columns from Hugging Face rows, drops verifiers and server
configuration, and measures whether a frozen LocalAgent dense selector can retrieve one of the
oracle-required tools from a distractor candidate set.  It is deliberately not an official
EnterpriseOps-Gym runner or task-success score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings
from localagent.agent.tool_head import _feat
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

ENTERPRISEOPSGYM_DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
ENTERPRISEOPSGYM_REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
ENTERPRISEOPSGYM_ADAPTER = "enterpriseopsgym-name-only-retrieval-v1"

_REQUIRED_ROW_KEYS = frozenset(
    {
        "task_id",
        "domain",
        "system_prompt",
        "user_prompt",
        "selected_tools",
    }
)


@dataclass(frozen=True)
class EnterpriseTask:
    """Sanitized public task fields used by the retrieval diagnostic."""

    task_id: str
    domain: str
    system_prompt: str
    user_prompt: str
    oracle_tools: tuple[str, ...]
    candidate_tools: tuple[str, ...]


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _tools(value: object, *, label: str, dedupe: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = tuple(_text(item, label=f"{label}[]") for item in value)
    if len(set(result)) != len(result) and not dedupe:
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(dict.fromkeys(result)) if dedupe else result


def _rows(path: str | Path) -> tuple[int, str, list[Mapping[str, Any]]]:
    source = Path(path)
    size, digest = _sha256(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid EnterpriseOps-Gym JSON rows file: {source}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
        raise ValueError("rows payload must contain a list under 'rows'")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(payload["rows"]):
        if not isinstance(item, Mapping) or not isinstance(item.get("row"), Mapping):
            raise ValueError(f"rows[{index}] must contain a row object")
        row = item["row"]
        missing = sorted(_REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise ValueError(f"rows[{index}] missing required keys: {missing}")
        result.append(row)
    if not result:
        raise ValueError("rows payload must not be empty")
    return size, digest, result


def load_tasks(oracle_path: str | Path, distractor_path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load matching oracle and distractor rows while dropping benchmark-only fields."""

    _, _, oracle_rows = _rows(oracle_path)
    _, _, distractor_rows = _rows(distractor_path)
    oracle = {_text(row["task_id"], label="oracle.task_id"): row for row in oracle_rows}
    distractors = {
        _text(row["task_id"], label="distractor.task_id"): row for row in distractor_rows
    }
    if set(oracle) != set(distractors):
        raise ValueError("oracle and distractor task IDs must match exactly")

    tasks: list[dict[str, Any]] = []
    for task_id in sorted(oracle):
        oracle_row = oracle[task_id]
        distractor_row = distractors[task_id]
        # Some rows preserve repeated calls in the candidate list.  Retrieval candidates are a
        # set, so remove repeats while retaining first-seen order; the raw source remains hash-bound.
        oracle_tools = _tools(
            oracle_row["selected_tools"], label=f"{task_id}.oracle_tools", dedupe=True
        )
        candidate_tools = _tools(
            distractor_row["selected_tools"],
            label=f"{task_id}.candidate_tools",
            dedupe=True,
        )
        if not set(oracle_tools).issubset(candidate_tools):
            raise ValueError(f"{task_id}: distractor candidates do not contain oracle tools")
        tasks.append(
            {
                "task": EnterpriseTask(
                    task_id=task_id,
                    domain=_text(oracle_row["domain"], label=f"{task_id}.domain"),
                    system_prompt=_text(oracle_row["system_prompt"], label=f"{task_id}.system_prompt"),
                    user_prompt=_text(oracle_row["user_prompt"], label=f"{task_id}.user_prompt"),
                    oracle_tools=oracle_tools,
                    candidate_tools=candidate_tools,
                ),
                "verifiers_dropped": True,
                "server_configuration_dropped": True,
            }
        )
    return tuple(tasks)


def _tool_rows(names: tuple[str, ...]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name=name,
            description=f"Enterprise tool: {name.replace('_', ' ')}.",
        )
        for name in names
    ]


@torch.no_grad()
def score_tasks(
    model: LocalAgentLM,
    tokenizer: Any,
    selector: DenseToolSelector,
    tasks: tuple[dict[str, Any], ...],
    *,
    device: str = "cpu",
) -> tuple[dict[str, Any], ...]:
    """Score retrieval against each task's oracle tool set without executing tools."""

    model.eval()
    selector.eval()
    scored: list[dict[str, Any]] = []
    for item in tasks:
        task = item["task"]
        tools = _tool_rows(task.candidate_tools)
        embeddings = tool_embeddings(tools, dim=selector.t_proj.in_features, device=device)
        prompt = f"{task.system_prompt}\nUser task: {task.user_prompt}"
        feature = _feat(model, tokenizer, prompt, device)
        scores = selector(feature.unsqueeze(0), embeddings)[0]
        order = torch.argsort(scores, descending=True).tolist()
        ranked = [task.candidate_tools[index] for index in order]
        gold = set(task.oracle_tools)
        scored.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "oracle_tool_count": len(gold),
                "candidate_tool_count": len(ranked),
                "top1": ranked[0],
                "top3": ranked[:3],
                "top5": ranked[:5],
                "hit_at_1": ranked[0] in gold,
                "hit_at_3": bool(set(ranked[:3]) & gold),
                "hit_at_5": bool(set(ranked[:5]) & gold),
            }
        )
    return tuple(scored)


def summarize_scores(scores: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Return aggregate retrieval metrics without retaining prompts or verifier content."""

    if not scores:
        raise ValueError("scores must not be empty")
    total = len(scores)
    summary: dict[str, Any] = {
        "records": total,
        "hit_at_1": sum(bool(row["hit_at_1"]) for row in scores) / total,
        "hit_at_3": sum(bool(row["hit_at_3"]) for row in scores) / total,
        "hit_at_5": sum(bool(row["hit_at_5"]) for row in scores) / total,
        "mean_oracle_tool_count": sum(row["oracle_tool_count"] for row in scores) / total,
        "mean_candidate_tool_count": sum(row["candidate_tool_count"] for row in scores) / total,
    }
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    summary["by_domain"] = {
        domain: {
            "records": len(rows),
            "hit_at_1": sum(bool(row["hit_at_1"]) for row in rows) / len(rows),
            "hit_at_3": sum(bool(row["hit_at_3"]) for row in rows) / len(rows),
            "hit_at_5": sum(bool(row["hit_at_5"]) for row in rows) / len(rows),
        }
        for domain, rows in sorted(by_domain.items())
    }
    return summary


def checkpoint_model(path: str | Path, *, device: str = "cpu") -> tuple[LocalAgentLM, Any, DenseToolSelector, dict[str, Any]]:
    """Load a frozen dispatch checkpoint and its recorded tokenizer."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    model = LocalAgentLM(ModelConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer_meta = checkpoint.get("tokenizer")
    if not isinstance(tokenizer_meta, Mapping):
        raise ValueError("checkpoint is missing tokenizer metadata")
    tokenizer = load_tokenizer(tokenizer_meta["kind"], tokenizer_meta.get("path"))
    dense_state = checkpoint.get("dense_selector")
    if not isinstance(dense_state, Mapping):
        raise ValueError("checkpoint is missing dense_selector state")
    q_weight = dense_state.get("q_proj.weight")
    t_weight = dense_state.get("t_proj.weight")
    if not isinstance(q_weight, torch.Tensor) or not isinstance(t_weight, torch.Tensor):
        raise ValueError("dense_selector checkpoint tensors are incomplete")
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=int(t_weight.shape[1]),
        proj=int(q_weight.shape[0]),
    ).to(device)
    selector.load_state_dict(dense_state)
    selector.eval()
    return model, tokenizer, selector, dict(checkpoint)


__all__ = [
    "ENTERPRISEOPSGYM_ADAPTER",
    "ENTERPRISEOPSGYM_DATASET",
    "ENTERPRISEOPSGYM_REVISION",
    "EnterpriseTask",
    "checkpoint_model",
    "load_tasks",
    "score_tasks",
    "summarize_scores",
]
