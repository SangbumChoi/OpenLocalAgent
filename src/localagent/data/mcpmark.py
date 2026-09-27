"""Leakage-safe profiling for the public MCPMark task metadata.

MCPMark is an execution benchmark with separate standard/easy suites and service-specific
verifiers.  This module reads only ``tasks/*/*/*/*/meta.json`` from a pinned checkout.  It keeps
task descriptions, state assets, and verifier code out of the profile and never emits a training
row or benchmark score.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MCPMARK_URL = "https://github.com/eval-sys/mcpmark"
MCPMARK_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"
MCPMARK_ADAPTER = "mcpmark_metadata_profile_v1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _list_text(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = tuple(_text(item, label=f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(result))


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _meta_paths(root: Path) -> tuple[Path, ...]:
    task_root = root / "tasks"
    if not task_root.is_dir():
        raise ValueError(f"MCPMark checkout is missing {task_root}")
    paths = tuple(sorted(task_root.glob("*/*/*/*/meta.json")))
    if not paths:
        raise ValueError("MCPMark checkout contains no task metadata files")
    return paths


def profile_mcpmark(root: str | Path, *, revision: str = MCPMARK_REVISION) -> dict[str, Any]:
    """Return a deterministic metadata inventory without retaining prompt text or verifiers."""

    checkout = Path(root)
    revision_text = _text(revision, label="revision")
    paths = _meta_paths(checkout)
    mcp_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    task_ids: list[str] = []

    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid MCPMark metadata: {path}") from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        relative_parts = path.relative_to(checkout).parts
        if len(relative_parts) != 6 or relative_parts[0] != "tasks":
            raise ValueError(f"unexpected MCPMark metadata path: {path}")
        _, service, suite, category_dir, task_dir, _ = relative_parts
        task_id = _text(raw.get("task_id"), label=f"{path}.task_id")
        category_id = _text(raw.get("category_id"), label=f"{path}.category_id")
        difficulty = _text(raw.get("difficulty"), label=f"{path}.difficulty")
        tags = _list_text(raw.get("tags"), label=f"{path}.tags")
        services = _list_text(raw.get("mcp"), label=f"{path}.mcp")
        metadata = raw.get("meta_data", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}.meta_data must be an object")
        state_type = metadata.get("stateType")
        state_counts[str(state_type) if state_type is not None else "none"] += 1
        file_size, file_sha = _sha256(path)
        task_ids.append(task_id)
        mcp_counts.update(services)
        backend_counts[service] += 1
        suite_counts[suite] += 1
        difficulty_counts[difficulty] += 1
        category_counts[category_id] += 1
        tag_counts.update(tags)
        rows.append(
            {
                "path": path.relative_to(checkout).as_posix(),
                "sha256": file_sha,
                "bytes": file_size,
                "service": service,
                "suite": suite,
                "category": category_id,
                "task_dir_sha256": hashlib.sha256(task_dir.encode("utf-8")).hexdigest(),
                "task_id_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
            }
        )
        # Keep these names in the path validation above so category/task layout cannot silently
        # drift, but never retain the upstream prompt or task name in the receipt.
        _ = category_dir

    canonical_rows = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    profile: dict[str, Any] = {
        "kind": "localagent_mcpmark_metadata_profile",
        "schema_version": 1,
        "dataset": "MCPMark",
        "dataset_url": MCPMARK_URL,
        "adapter": MCPMARK_ADAPTER,
        "source_revision": revision_text,
        "source": {
            "checkout_root": "external_checkout",
            "metadata_glob": "tasks/*/*/*/*/meta.json",
            "metadata_files": len(rows),
            "metadata_bytes": sum(row["bytes"] for row in rows),
            "metadata_manifest_sha256": hashlib.sha256(canonical_rows).hexdigest(),
            "task_id_set_sha256": hashlib.sha256("\n".join(sorted(task_ids)).encode("utf-8")).hexdigest(),
            "description_text_retained": False,
            "state_assets_consumed": False,
            "verifiers_consumed": False,
        },
        "mcp_service_counts": dict(sorted(mcp_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        "backend_directory_counts": dict(
            sorted(backend_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        "suite_counts": dict(sorted(suite_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "category_counts_top20": dict(
            sorted(category_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:20]
        ),
        "tag_counts_top20": dict(
            sorted(tag_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:20]
        ),
        "state_type_counts": dict(sorted(state_counts.items())),
        "realistic_surface_counts": {
            "notion_tasks": mcp_counts.get("notion", 0),
            "browser_tasks": mcp_counts.get("playwright_webarena", 0)
            + mcp_counts.get("playwright", 0),
            "filesystem_tasks": mcp_counts.get("filesystem", 0),
            "github_tasks": mcp_counts.get("github", 0),
            "database_tasks": sum(mcp_counts.get(name, 0) for name in ("postgres", "supabase", "insforge")),
        },
        "claim_boundary": (
            "Metadata inventory only; no prompt text, trajectories, state assets, verifier code, "
            "or MCPMark model score was retained. Standard and easy tasks remain evaluation-only "
            "under LocalAgent's contamination policy."
        ),
    }
    profile["profile_sha256"] = hashlib.sha256(
        json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return profile


__all__ = ["MCPMARK_ADAPTER", "MCPMARK_REVISION", "MCPMARK_URL", "profile_mcpmark"]
