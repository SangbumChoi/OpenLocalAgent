"""Leakage-safe MCPMark task-to-service routing evaluation.

MCPMark's official score requires live MCP servers, state fixtures, and verifiers.  This adapter
uses only the public task metadata and descriptions to measure whether a checkpoint routes a
realistic task to the right service/tool family.  It never retains task text in the receipt and
must not be reported as an MCPMark execution score.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector
from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
from localagent.agent.routes import ROUTES, RouteHead
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer

MCPMARK_URL = "https://github.com/eval-sys/mcpmark"
_ALIASES = {"playwright_webarena": "playwright", "supabase": "postgres", "insforge": "postgres"}

# These are intentionally sets rather than single gold tools: a task description can require
# several calls.  The metric asks whether the checkpoint stayed within the correct service family.
SERVICE_TOOL_FAMILIES: dict[str, frozenset[str]] = {
    "filesystem": frozenset(
        {"read_file", "write_file", "list_dir", "find_files", "make_dir", "unzip", "grep_search"}
    ),
    "github": frozenset(
        {"http_request", "run_command", "git_diff", "git_status", "git_commit", "read_file", "write_file", "apply_patch"}
    ),
    "notion": frozenset({"notion_create_page", "notion_write"}),
    "playwright": frozenset(
        {"open_url", "screenshot", "click", "double_click", "type_text", "key_press", "scroll", "drag", "wait", "move_cursor"}
    ),
    "postgres": frozenset({"sql_query"}),
}

SERVICE_ROUTES: dict[str, frozenset[str]] = {
    "filesystem": frozenset({"code"}),
    "github": frozenset({"code", "web_search"}),
    "notion": frozenset({"app_action"}),
    "playwright": frozenset({"computer_use", "web_search"}),
    "postgres": frozenset({"code"}),
}


def _canonical_service(name: str) -> str:
    return _ALIASES.get(name, name)


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _discover(root: Path, suite: str) -> list[tuple[str, Path, Path]]:
    task_root = root / "tasks"
    rows: list[tuple[str, Path, Path]] = []
    for meta_path in sorted(task_root.glob(f"*/{suite}/*/*/meta.json")):
        parts = meta_path.relative_to(task_root).parts
        if len(parts) != 5:
            raise ValueError(f"unexpected MCPMark task path: {meta_path}")
        service = _canonical_service(parts[0])
        description_path = meta_path.with_name("description.md")
        if service not in SERVICE_TOOL_FAMILIES:
            raise ValueError(f"unsupported MCPMark service {service!r}: {meta_path}")
        if not description_path.is_file():
            raise ValueError(f"MCPMark task is missing description.md: {description_path}")
        rows.append((service, meta_path, description_path))
    if not rows:
        raise ValueError(f"no MCPMark {suite!r} task metadata found under {task_root}")
    return rows


def _checkpoint_tokenizer(checkpoint: Mapping[str, Any]):
    metadata = checkpoint.get("tokenizer")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint is missing tokenizer metadata")
    kind = str(metadata.get("kind", "byte"))
    path = metadata.get("path")
    if kind == "bpe" and not isinstance(path, str):
        raise ValueError("BPE checkpoint is missing tokenizer.path")
    tokenizer = load_tokenizer(kind, path)
    cfg = checkpoint.get("cfg")
    vocab_size = cfg.get("vocab_size") if isinstance(cfg, Mapping) else None
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise ValueError("checkpoint tokenizer vocabulary does not match model config")
    return tokenizer


def _source_manifest(rows: list[tuple[str, Path, Path]]) -> tuple[int, str]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for service, meta_path, description_path in rows:
        for path in (meta_path, description_path):
            size, digest = _sha256(path)
            total_bytes += size
            records.append(
                {
                    "bytes": size,
                    "path": str(path),
                    "service": service,
                    "sha256": digest,
                }
            )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return total_bytes, hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def _features(model: LocalAgentLM, tokenizer, prompts: list[str], device: str) -> torch.Tensor:
    encoded = [tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")[-model.cfg.max_seq_len :] for prompt in prompts]
    width = max(len(row) for row in encoded)
    inputs = torch.full((len(encoded), width), tokenizer.pad_id, dtype=torch.long, device=device)
    positions: list[int] = []
    for index, row in enumerate(encoded):
        inputs[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        positions.append(len(row) - 1)
    _, hidden = model(inputs, return_hidden=True)
    return torch.stack([hidden[index, position] for index, position in enumerate(positions)])


def evaluate_mcpmark_router(
    checkout: str | Path,
    checkpoint: str | Path,
    *,
    suite: str = "standard",
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate service-family routing over public MCPMark task descriptions."""

    root = Path(checkout).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    rows = _discover(root, suite)
    source_bytes, source_manifest_sha256 = _source_manifest(rows)
    source_revision = _git_head(root)
    checkpoint_bytes, checkpoint_sha256 = _sha256(checkpoint_path)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    cfg = ModelConfig(**raw_checkpoint["cfg"])
    model = LocalAgentLM(cfg).to(device).eval()
    model.load_state_dict(raw_checkpoint["state_dict"])
    tokenizer = _checkpoint_tokenizer(raw_checkpoint)
    tools = list(STANDARD_TOOLS) + mobile_tools() + realistic_productivity_tools()
    selector_state = raw_checkpoint.get("dense_selector")
    route_state = raw_checkpoint.get("route_head")
    if not isinstance(selector_state, Mapping) or not isinstance(route_state, Mapping):
        raise ValueError("checkpoint must contain route_head and dense_selector")
    selector = DenseToolSelector(
        cfg.d_model,
        emb_dim=int(raw_checkpoint.get("selector_proj", 256)) * 32,
        proj=int(raw_checkpoint.get("selector_proj", 256)),
    ).to(device)
    # The serialized selector exposes its true input width through the first t_proj tensor.
    t_proj_weight = selector_state.get("t_proj.weight")
    if isinstance(t_proj_weight, torch.Tensor):
        selector = DenseToolSelector(
            cfg.d_model,
            emb_dim=int(t_proj_weight.shape[1]),
            proj=int(t_proj_weight.shape[0]),
        ).to(device)
    selector.load_state_dict(selector_state)
    selector.eval()
    route = RouteHead(cfg.d_model).to(device)
    route.load_state_dict(route_state)
    route.eval()
    bound = BoundSelector(selector, tools, device=device, examples=raw_checkpoint.get("examples"))

    prompts: list[str] = []
    services: list[str] = []
    difficulties: list[str] = []
    for service, meta_path, description_path in rows:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid MCPMark metadata: {meta_path}")
        description = description_path.read_text(encoding="utf-8").strip()
        prompts.append(
            f"MCP service: {service}\nCategory: {metadata.get('category_name', '')}\n"
            f"Task: {metadata.get('description', '')}\nInstructions:\n{description}"
        )
        services.append(service)
        difficulties.append(str(metadata.get("difficulty", "")))
    features = _features(model, tokenizer, prompts, device)
    by_service: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_difficulty: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = {"rows": len(rows), "route_correct": 0, "selector_top1": 0, "selector_top3": 0}
    for feature, service, difficulty in zip(features, services, difficulties):
        predicted_route = ROUTES[int(route(feature).argmax(-1))]
        ranked = bound.rank(feature)
        expected_tools = SERVICE_TOOL_FAMILIES[service]
        expected_routes = SERVICE_ROUTES[service]
        route_correct = int(predicted_route in expected_routes)
        top1 = int(ranked[0] in expected_tools)
        top3 = int(bool(set(ranked[:3]) & expected_tools))
        total["route_correct"] += route_correct
        total["selector_top1"] += top1
        total["selector_top3"] += top3
        for bucket, key in ((by_service, service), (by_difficulty, difficulty)):
            bucket[key]["rows"] += 1
            bucket[key]["route_correct"] += route_correct
            bucket[key]["selector_top1"] += top1
            bucket[key]["selector_top3"] += top3

    def finalize(values: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
        return {
            key: {
                **stats,
                "route_accuracy": stats["route_correct"] / max(1, stats["rows"]),
                "selector_top1": stats["selector_top1"] / max(1, stats["rows"]),
                "selector_top3": stats["selector_top3"] / max(1, stats["rows"]),
            }
            for key, stats in sorted(values.items())
        }

    return {
        "kind": "localagent_mcpmark_task_router_proxy",
        "schema_version": 1,
        "dataset": {"url": MCPMARK_URL, "revision": source_revision, "suite": suite},
        "source": {
            "checkout": str(root),
            "task_rows": len(rows),
            "metadata_and_description_bytes": source_bytes,
            "task_manifest_sha256": source_manifest_sha256,
            "task_text_retained": False,
            "verifiers_executed": False,
            "mcp_servers_executed": False,
        },
        "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_bytes, "sha256": checkpoint_sha256},
        "tool_family_contract": {key: sorted(value) for key, value in sorted(SERVICE_TOOL_FAMILIES.items())},
        "overall": {
            **total,
            "route_accuracy": total["route_correct"] / max(1, total["rows"]),
            "selector_top1": total["selector_top1"] / max(1, total["rows"]),
            "selector_top3": total["selector_top3"] / max(1, total["rows"]),
        },
        "by_service": finalize(by_service),
        "by_difficulty": finalize(by_difficulty),
        "claim_boundary": "Public MCPMark task-description service-routing proxy only; not live MCP execution, verifier success, pass@k, or official leaderboard evaluation.",
    }


__all__ = ["MCPMARK_URL", "SERVICE_ROUTES", "SERVICE_TOOL_FAMILIES", "evaluate_mcpmark_router"]
