#!/usr/bin/env python3
"""Audit MCP-Persona's public task and simulator release without running MCP servers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = "MCP-Persona"
SOURCE_URL = "https://github.com/wwh0411/MCP-Persona"
PAPER_URL = "https://arxiv.org/abs/2606.02470"
REVISION = "b510f5a5371c4524a58aeeb679c1ace845603e95"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("MCP-Persona release must be a JSON list of task objects")
    return rows


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    servers = Counter(str(tool).split(":", 1)[0] for row in rows for tool in row.get("chains", []))
    tools = {str(tool) for row in rows for tool in row.get("chains", [])}
    checkpoints = Counter(
        str(checkpoint.get("checkpoint_type"))
        for row in rows
        for checkpoint in row.get("gt", [])
        if isinstance(checkpoint, dict)
    )
    return {
        "tasks": len(rows),
        "unique_ids": len({row.get("id") for row in rows}),
        "id_min": min(int(row["id"]) for row in rows),
        "id_max": max(int(row["id"]) for row in rows),
        "query_types": dict(sorted(Counter(str(row.get("query_type")) for row in rows).items())),
        "chain_lengths": {
            str(length): count
            for length, count in sorted(Counter(len(row.get("chains", [])) for row in rows).items())
        },
        "servers": dict(sorted(servers.items())),
        "unique_tools": len(tools),
        "checkpoint_types": dict(sorted(checkpoints.items())),
        "tasks_without_gt": sum(not row.get("gt") for row in rows),
        "tasks_without_gt_annotation": sum(not row.get("gt_annotation") for row in rows),
    }


def audit(repo: Path) -> dict[str, Any]:
    en_path = repo / "data/tasks/en_release_data.json"
    zh_path = repo / "data/tasks/zh_release_data.json"
    readme_path = repo / "README.md"
    license_path = repo / "LICENSE"
    en_rows = _load_rows(en_path)
    zh_rows = _load_rows(zh_path)
    release_ids = {int(row["id"]) for row in en_rows}
    zh_ids = {int(row["id"]) for row in zh_rows}
    simulated_root = repo / "data/simulated_tools"
    simulated_servers = sorted(path.name for path in simulated_root.iterdir() if path.is_dir())
    chain_servers = sorted({str(tool).split(":", 1)[0] for row in en_rows for tool in row.get("chains", [])})
    body: dict[str, Any] = {
        "kind": "localagent_mcp_persona_source_audit",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "source_url": SOURCE_URL,
            "paper_url": PAPER_URL,
            "revision": REVISION,
            "files": {
                "readme": _identity(readme_path),
                "license": _identity(license_path) if license_path.exists() else None,
                "english_tasks": _identity(en_path),
                "chinese_tasks": _identity(zh_path),
            },
            "license_evidence": {
                "declared_by_readme_badge": "MIT",
                "license_file_present": license_path.exists(),
                "status": "metadata_badge_only" if not license_path.exists() else "file_present",
            },
        },
        "releases": {"english": _profile(en_rows), "chinese": _profile(zh_rows)},
        "cross_release": {
            "english_chinese_task_id_overlap": len(release_ids & zh_ids),
            "english_chinese_ids_match": release_ids == zh_ids,
        },
        "runtime_inventory": {
            "task_chain_servers": chain_servers,
            "static_simulated_server_directories": simulated_servers,
            "chain_servers_without_static_simulator": sorted(set(chain_servers) - set(simulated_servers)),
            "static_simulator_coverage_count": len(set(chain_servers) & set(simulated_servers)),
        },
        "evaluation_boundary": {
            "public_tasks": True,
            "public_simulated_tools": True,
            "official_train_test_split": False,
            "train_policy": "eval_only",
            "reason": (
                "The release contains English and Chinese copies of the same 173-task benchmark, "
                "not a train/test partition. Tasks include ground-truth personalized-search and "
                "operate checkpoints; the repository ships only a subset of static simulators while "
                "the remaining server/tool behavior is generated or configured upstream."
            ),
        },
        "claim_boundary": (
            "Source/protocol audit only. No MCP server, simulated context, model checkpoint, LLM judge, "
            "credential, or external side effect was executed; no MCP-Persona score is claimed."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = audit(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
