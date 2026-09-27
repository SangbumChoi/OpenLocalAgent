"""Frozen-backbone API-schema head for the bounded AppWorld adapter.

This is deliberately separate from the deployment tool heads.  It classifies a prompt into an
``app.api`` label using frozen LocalAgent features, so a native evaluator can restrict schema
grounding to the API family learned from public train rows.  It is an adapter diagnostic, not an
official AppWorld policy or a replacement for multi-step program generation.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.agent.retriever import embed

_API_CODE_RE = re.compile(r"^apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\(")


class AppWorldAPIHead(nn.Module):
    """A compact linear classifier over frozen model prompt features."""

    def __init__(self, d_model: int, classes: Iterable[str]):
        super().__init__()
        labels = tuple(str(label) for label in classes)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("AppWorld API head classes must be non-empty and unique")
        self.classes = labels
        self.fc = nn.Linear(d_model, len(labels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class AppWorldAPINearestNeighbor:
    """Prompt-side char-ngram retrieval adapter over public train examples.

    This intentionally has no learned model weights.  It is a schema-grounding control that
    measures how much of the remaining AppWorld gap is lexical retrieval rather than the
    pretrained representation.
    """

    def __init__(
        self,
        prompts: Iterable[str],
        labels: Iterable[str],
        dim: int = 8192,
        argument_fields: dict[str, Iterable[str]] | None = None,
    ):
        self.prompts = tuple(str(prompt) for prompt in prompts)
        self.labels = tuple(str(label) for label in labels)
        if not self.prompts or len(self.prompts) != len(self.labels):
            raise ValueError("retriever prompts and labels must be non-empty and equally sized")
        self.dim = int(dim)
        matrix = torch.from_numpy(
            np.stack([embed(prompt, dim=self.dim) for prompt in self.prompts])
        ).to(dtype=torch.float32)
        self.matrix = F.normalize(matrix, dim=-1)
        self.classes = tuple(sorted(set(self.labels)))
        self.argument_fields = {
            str(label): tuple(sorted(str(field) for field in fields))
            for label, fields in (argument_fields or {}).items()
        }

    def predict(self, prompt: str) -> str:
        query = torch.tensor(embed(prompt, dim=self.dim), dtype=torch.float32)
        query = F.normalize(query.unsqueeze(0), dim=-1)
        index = int((query @ self.matrix.T).argmax(-1).item())
        return self.labels[index]


def api_label_from_code(code: str) -> str:
    """Extract the canonical ``app.api`` label from one literal AppWorld API expression."""

    match = _API_CODE_RE.match(code.strip())
    if match is None:
        raise ValueError(f"unsupported AppWorld API code: {code!r}")
    return f"{match.group(1)}.{match.group(2)}"


def first_action_examples(rows: Iterable[Any]) -> tuple[list[str], list[str]]:
    """Return prompt/API-label pairs from normalized first-action Conversation rows."""

    prompts: list[str] = []
    labels: list[str] = []
    for row in rows:
        prompt = ""
        code: str | None = None
        for message in row.messages:
            if getattr(message, "role", None).value == "user" and not prompt:
                prompt = str(message.content)
            for call in getattr(message, "tool_calls", []):
                if call.name == "run_python" and isinstance(call.arguments.get("code"), str):
                    code = call.arguments["code"]
                    break
            if code is not None:
                break
        if not prompt or code is None:
            raise ValueError("each AppWorld row needs one user prompt and run_python code")
        prompts.append(prompt)
        labels.append(api_label_from_code(code))
    if not prompts:
        raise ValueError("no AppWorld API examples found")
    return prompts, labels


def first_action_argument_fields(rows: Iterable[Any]) -> dict[str, set[str]]:
    """Collect non-credential keyword fields observed for each public train API."""

    fields: dict[str, set[str]] = {}
    for row in rows:
        code: str | None = None
        for message in row.messages:
            for call in getattr(message, "tool_calls", []):
                if call.name == "run_python" and isinstance(call.arguments.get("code"), str):
                    code = call.arguments["code"]
                    break
            if code is not None:
                break
        if code is None:
            continue
        try:
            tree = ast.parse(code, mode="exec")
            call = tree.body[0].value
            label = api_label_from_code(code)
            fields.setdefault(label, set()).update(
                keyword.arg
                for keyword in call.keywords
                if keyword.arg is not None and keyword.arg != "access_token"
            )
        except (AttributeError, IndexError, SyntaxError, TypeError):
            continue
    return fields


def _features(model: Any, tokenizer: Any, prompts: list[str], device: str) -> torch.Tensor:
    from localagent.agent.tool_head import _feat

    model.eval()
    with torch.no_grad():
        return torch.stack([_feat(model, tokenizer, prompt, device, framed=False) for prompt in prompts])


def head_metrics(head: AppWorldAPIHead, features: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    label_index = {name: index for index, name in enumerate(head.classes)}
    if any(label not in label_index for label in labels):
        raise ValueError("evaluation contains an API class absent from the train head")
    targets = torch.tensor([label_index[label] for label in labels], device=features.device)
    with torch.no_grad():
        predictions = head(features).argmax(-1)
    correct = int((predictions == targets).sum().item())
    return {"rows": len(labels), "exact": correct, "accuracy": correct / max(1, len(labels))}


def train_appworld_api_head(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    labels: list[str],
    *,
    steps: int = 256,
    batch_size: int = 32,
    lr: float = 5e-3,
    device: str = "cpu",
    seed: int = 2027,
    log=lambda *args: None,
) -> tuple[AppWorldAPIHead, dict[str, Any]]:
    """Train the API head on detached frozen-body features and return train metrics."""

    if len(prompts) != len(labels) or not prompts:
        raise ValueError("prompts and labels must be non-empty and equally sized")
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("steps, batch_size, and lr must be positive")
    classes = tuple(sorted(set(labels)))
    features = _features(model, tokenizer, prompts, device)
    label_index = {name: index for index, name in enumerate(classes)}
    targets = torch.tensor([label_index[label] for label in labels], device=device)
    head = AppWorldAPIHead(model.cfg.d_model, classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for step in range(steps):
        indices = torch.randint(
            len(labels), (batch_size,), generator=generator, device=device
        )
        loss = F.cross_entropy(head(features[indices]), targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            metrics = head_metrics(head, features, labels)
            log(f"  [appworld-api-head] step {step}/{steps} loss {loss.item():.3f} acc {metrics['accuracy']:.3f}")
    return head, head_metrics(head, features, labels)


def save_appworld_api_head(
    path: Path,
    head: AppWorldAPIHead,
    *,
    parent_checkpoint: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Serialize an API head with enough lineage to reject incompatible checkpoints."""

    payload = {
        "kind": "localagent_appworld_api_head",
        "schema_version": 1,
        "d_model": int(head.fc.in_features),
        "classes": list(head.classes),
        "state_dict": head.state_dict(),
        "parent_checkpoint": parent_checkpoint,
        "source": source,
    }
    torch.save(payload, path)
    return payload


def load_appworld_api_head(path: Path, *, d_model: int) -> AppWorldAPIHead:
    """Load and validate a serialized API head for a model width."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") == "localagent_appworld_api_retriever":
        return AppWorldAPINearestNeighbor(
            payload.get("prompts", []),
            payload.get("labels", []),
            dim=int(payload.get("embedding_dim", 8192)),
            argument_fields=payload.get("argument_fields"),
        )
    if payload.get("kind") != "localagent_appworld_api_head":
        raise ValueError("invalid AppWorld API head kind")
    if int(payload.get("d_model", -1)) != d_model:
        raise ValueError("AppWorld API head d_model does not match checkpoint")
    head = AppWorldAPIHead(d_model, payload.get("classes", []))
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head
