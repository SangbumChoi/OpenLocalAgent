"""Leakage-safe profiling for the public Toolathlon-GYM task configurations.

Toolathlon-GYM is an execution benchmark, not a LocalAgent training corpus.  This module reads
only ``tasks/finalpool/*/task_config.json`` from a checked-out upstream revision.  It deliberately
does not open task descriptions, preprocessors, evaluators, initial workspaces, or ground-truth
outputs.  The resulting aggregate profile is useful for choosing realistic email/Notion/browser
coverage without copying benchmark prompts or verifier logic into training data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TOOLATHLON_GYM_URL = "https://github.com/eigent-ai/toolathlon_gym"
TOOLATHLON_GYM_REVISION = "45bbff81419fedbb6d5a2cb46029d2980a7c93c4"
TOOLATHLON_GYM_ADAPTER = "toolathlon_gym_config_profile_v1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _servers(value: object, *, label: str) -> tuple[str, ...]:
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


def _config_paths(root: Path) -> tuple[Path, ...]:
    task_root = root / "tasks" / "finalpool"
    if not task_root.is_dir():
        raise ValueError(f"Toolathlon-GYM checkout is missing {task_root}")
    paths = tuple(sorted(task_root.glob("*/task_config.json")))
    if not paths:
        raise ValueError("Toolathlon-GYM checkout contains no finalpool task_config.json files")
    return paths


def profile_toolathlon_gym(
    root: str | Path,
    *,
    revision: str = TOOLATHLON_GYM_REVISION,
) -> dict[str, Any]:
    """Return a deterministic aggregate profile without consuming benchmark payloads."""

    checkout = Path(root)
    revision_text = _text(revision, label="revision")
    paths = _config_paths(checkout)
    server_counts: Counter[str] = Counter()
    local_tool_counts: Counter[str] = Counter()
    server_count_distribution: Counter[str] = Counter()
    server_pairs: Counter[str] = Counter()
    config_rows: list[dict[str, Any]] = []
    task_names: list[str] = []

    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Toolathlon-GYM config: {path}") from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        servers = _servers(raw.get("needed_mcp_servers"), label=f"{path}.needed_mcp_servers")
        local_tools_value = raw.get("needed_local_tools", [])
        if not isinstance(local_tools_value, list):
            raise ValueError(f"{path}.needed_local_tools must be a list")
        local_tools = tuple(
            sorted(_text(item, label=f"{path}.needed_local_tools[]") for item in local_tools_value)
        )
        if len(set(local_tools)) != len(local_tools):
            raise ValueError(f"{path}.needed_local_tools must not contain duplicates")
        meta = raw.get("meta", {})
        if not isinstance(meta, Mapping):
            raise ValueError(f"{path}.meta must be an object")
        relative = path.relative_to(checkout).as_posix()
        _, config_sha = _sha256(path)
        task_names.append(path.parent.name)
        server_counts.update(servers)
        local_tool_counts.update(local_tools)
        server_count_distribution[str(len(servers))] += 1
        for left_index, left in enumerate(servers):
            for right in servers[left_index + 1 :]:
                server_pairs[f"{left}+{right}"] += 1
        config_rows.append(
            {
                "path": relative,
                "sha256": config_sha,
                "mcp_servers": list(servers),
                "local_tool_count": len(local_tools),
                "meta_keys": sorted(str(key) for key in meta),
            }
        )

    canonical_rows = json.dumps(
        config_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    task_name_digest = hashlib.sha256("\n".join(sorted(task_names)).encode("utf-8")).hexdigest()
    profile: dict[str, Any] = {
        "kind": "localagent_toolathlon_gym_config_profile",
        "schema_version": 1,
        "dataset": "Toolathlon-GYM",
        "dataset_url": TOOLATHLON_GYM_URL,
        "adapter": TOOLATHLON_GYM_ADAPTER,
        "source_revision": revision_text,
        "source": {
            "checkout_root": "external_checkout",
            "config_glob": "tasks/finalpool/*/task_config.json",
            "config_files": len(config_rows),
            "config_bytes": sum(_sha256(path)[0] for path in paths),
            "config_manifest_sha256": hashlib.sha256(canonical_rows).hexdigest(),
            "task_name_sha256": task_name_digest,
            "descriptions_consumed": False,
            "preprocessors_consumed": False,
            "evaluators_consumed": False,
            "workspaces_consumed": False,
        },
        "mcp_server_counts": dict(sorted(server_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        "local_tool_counts": dict(
            sorted(local_tool_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        "task_server_count_distribution": dict(sorted(server_count_distribution.items())),
        "server_pair_counts_top20": dict(
            sorted(server_pairs.items(), key=lambda pair: (-pair[1], pair[0]))[:20]
        ),
        "realistic_surface_counts": {
            "email_tasks": server_counts.get("emails", 0),
            "notion_tasks": server_counts.get("notion", 0),
            "browser_playwright_tasks": server_counts.get("playwright_with_chunk", 0),
            "filesystem_tasks": server_counts.get("filesystem", 0),
            "calendar_tasks": server_counts.get("google_calendar", 0),
        },
        "claim_boundary": (
            "Configuration inventory only; no task descriptions, trajectories, workspaces, "
            "ground-truth outputs, evaluator code, or model score was consumed. Toolathlon-GYM "
            "tasks remain evaluation-only under LocalAgent's contamination policy."
        ),
    }
    profile["profile_sha256"] = hashlib.sha256(
        json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return profile


__all__ = [
    "TOOLATHLON_GYM_ADAPTER",
    "TOOLATHLON_GYM_REVISION",
    "TOOLATHLON_GYM_URL",
    "profile_toolathlon_gym",
]
