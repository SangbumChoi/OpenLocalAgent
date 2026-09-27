#!/usr/bin/env python3
"""Re-bind ToolSandbox public projection rows to the pinned runtime ToolSpec contract.

The existing AST projection is useful for source-level schema learning, but its candidate lists
are not necessarily the same lists that the live simulator exposes.  This adapter imports only
the upstream scenario/tool definitions (it does not play scenarios, execute tools, load seeded
state, or call an API), converts the exact agent-visible runtime tools through ToolSandbox's own
OpenAI conversion helper, and rewrites each row's candidate catalog.  The resulting Conversation
JSONL remains a projection diagnostic, not an official ToolSandbox dataset or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, ToolSpec

SOURCE_URL = "https://github.com/apple/ToolSandbox"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
SEED = 211


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load_rows(path: Path) -> list[Conversation]:
    rows = [Conversation.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty projection: {path}")
    return rows


def _runtime_catalog(root: Path) -> dict[str, list[ToolSpec]]:
    random.seed(SEED)
    sys.path.insert(0, str(root.resolve()))
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.common.tool_conversion import convert_to_openai_tool
    from tool_sandbox.scenarios import named_scenarios

    scenarios = named_scenarios(ToolBackend.DEFAULT)
    catalogs: dict[str, list[ToolSpec]] = {}
    for name, scenario in scenarios.items():
        available = scenario.starting_context.get_available_tools(scrambling_allowed=False)
        specs: list[ToolSpec] = []
        for tool_name, function in sorted(available.items()):
            declaration = convert_to_openai_tool(function, name=tool_name)["function"]
            specs.append(
                ToolSpec(
                    name=str(declaration["name"]),
                    description=str(declaration.get("description", "")),
                    parameters=dict(declaration.get("parameters", {})),
                )
            )
        catalogs[name] = specs
    if not catalogs:
        raise ValueError("ToolSandbox runtime catalog is empty")
    return catalogs


def _rewrite(
    rows: list[Conversation], catalogs: dict[str, list[ToolSpec]]
) -> tuple[list[Conversation], list[dict[str, Any]]]:
    rewritten: list[Conversation] = []
    dropped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        provenance = row.meta.get("provenance", {})
        scenario_name = provenance.get("scenario_name")
        if not isinstance(scenario_name, str) or scenario_name not in catalogs:
            raise ValueError(f"row {index} has unknown scenario_name: {scenario_name!r}")
        target_names = {
            call.name
            for message in row.messages
            for call in message.tool_calls
        }
        runtime_names = {tool.name for tool in catalogs[scenario_name]}
        missing = sorted(target_names - runtime_names)
        if missing:
            dropped.append(
                {
                    "row_index": index,
                    "scenario_name": scenario_name,
                    "target_tools": sorted(target_names),
                    "missing_runtime_tools": missing,
                    "reason": "static_projection_target_not_available_in_runtime_context",
                }
            )
            continue
        meta = dict(row.meta)
        meta.update(
            {
                "derivation": "runtime_toolspec_projection_v1",
                "runtime_catalog_source": "ToolSandbox.ExecutionContext.get_available_tools",
                "runtime_scenario_name": scenario_name,
                "runtime_tool_count": len(catalogs[scenario_name]),
                "runtime_state_history_available": False,
                "runtime_simulator_executed": False,
                "runtime_verifier_executed": False,
            }
        )
        rewritten.append(Conversation(messages=row.messages, tools=catalogs[scenario_name], meta=meta))
    return rewritten, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite runtime projection outputs")
    if not args.source_root.is_dir():
        raise SystemExit(f"missing ToolSandbox source root: {args.source_root}")
    rows = _load_rows(args.input)
    catalogs = _runtime_catalog(args.source_root)
    rewritten, dropped = _rewrite(rows, catalogs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(row.to_json() + "\n" for row in rewritten), encoding="utf-8")
    manifest = {
        "kind": "localagent_toolsandbox_runtime_projection",
        "schema_version": 1,
        "source": {
            "name": "apple/ToolSandbox",
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "root": str(args.source_root.resolve()),
        },
        "input": _identity(args.input),
        "output": _identity(args.output),
        "input_rows": len(rows),
        "rows": len(rewritten),
        "dropped_rows": len(dropped),
        "dropped_examples": dropped[:32],
        "scenario_count": len(catalogs),
        "tool_catalog_size": len({tool.name for tools in catalogs.values() for tool in tools}),
        "runtime_catalog_seed": SEED,
        "state_history_available": False,
        "environment_executed": False,
        "verifier_executed": False,
        "claim_boundary": (
            "Runtime ToolSpec and initial candidate-catalog projection only. ToolSandbox scenarios, "
            "simulator state, user simulator, verifiers, external APIs, and official split were not executed."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
