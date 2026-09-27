#!/usr/bin/env python
"""Profile Apple ToolSandbox scenario source without importing or running it.

ToolSandbox is a stateful simulator rather than a static prompt file.  This command reads only
the public Python source and README, extracts literal scenario/category/tool metadata with the
standard-library AST, and binds the inventory to an upstream commit.  It never imports
ToolSandbox, executes a scenario, starts its user simulator, calls an external API, or retains
scenario prompts as training data.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "apple/ToolSandbox"
SOURCE_URL = "https://github.com/apple/ToolSandbox"
REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
LICENSE = "Apple Software License; see upstream LICENSE and ACKNOWLEDGEMENTS"

SCENARIO_FILES = (
    "single_tool_call_scenarios.py",
    "multiple_tool_call_scenarios.py",
    "multiple_user_turn_scenarios.py",
    "insufficient_information_scenarios.py",
)
REQUIRED_SOURCE_FILES = SCENARIO_FILES + (
    "../common/execution_context.py",
    "../scenarios/__init__.py",
    "../../README.md",
    "../../LICENSE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _scenario_name_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "name" and _literal_string(keyword.value):
                calls.append(node)
                break
    return calls


def _categories(call: ast.Call) -> list[str]:
    values: list[str] = []
    for keyword in call.keywords:
        if keyword.arg != "categories":
            continue
        for node in ast.walk(keyword.value):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "ScenarioCategories"
            ):
                values.append(node.attr)
    return values


def _tool_names(call: ast.Call) -> list[str]:
    for keyword in call.keywords:
        if keyword.arg != "tool_allow_list" or not isinstance(keyword.value, (ast.List, ast.Tuple)):
            continue
        values = [_literal_string(item) for item in keyword.value.elts]
        return [value for value in values if value is not None]
    return []


def _counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(values).items(), key=lambda pair: (-pair[1], pair[0])))


def _source_entry(root: Path, relative: str) -> dict[str, Any]:
    path = (root / "tool_sandbox/scenarios" / relative).resolve()
    if relative.startswith("../common"):
        path = (root / "tool_sandbox/scenarios" / relative).resolve()
    elif relative.startswith("../../"):
        path = (root / "tool_sandbox/scenarios" / relative).resolve()
    if not path.is_file():
        raise ValueError(f"required ToolSandbox source is missing: {path}")
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def profile(root: Path, *, revision: str = REVISION) -> dict[str, Any]:
    """Return a deterministic inventory of ToolSandbox's literal scenario definitions."""

    scenario_root = root / "tool_sandbox/scenarios"
    if not scenario_root.is_dir():
        raise ValueError(f"ToolSandbox checkout is missing scenario directory: {scenario_root}")

    family_rows: dict[str, dict[str, Any]] = {}
    all_names: list[str] = []
    all_categories: list[str] = []
    all_tools: list[str] = []
    total_tool_lists = 0
    for filename in SCENARIO_FILES:
        path = scenario_root / filename
        if not path.is_file():
            raise ValueError(f"required ToolSandbox scenario file is missing: {path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise ValueError(f"unable to parse ToolSandbox scenario source: {path}") from error

        names: list[str] = []
        categories: list[str] = []
        tools: list[str] = []
        tool_lists = 0
        for call in _scenario_name_calls(tree):
            name = next(
                _literal_string(keyword.value)
                for keyword in call.keywords
                if keyword.arg == "name"
            )
            assert name is not None
            names.append(name)
            categories.extend(_categories(call))
            call_tools = _tool_names(call)
            if call_tools:
                tool_lists += 1
                tools.extend(call_tools)
        if not names:
            raise ValueError(f"no literal scenario names found in {path}")
        family = filename.removesuffix("_scenarios.py")
        family_rows[family] = {
            "base_scenarios": len(names),
            "unique_base_scenarios": len(set(names)),
            "category_tokens": _counter(categories),
            "tool_allow_list_literals": tool_lists,
            "tool_allow_list_entries": len(tools),
            "unique_tools": len(set(tools)),
            "scenario_name_sha256": hashlib.sha256(
                "\n".join(names).encode("utf-8")
            ).hexdigest(),
        }
        all_names.extend(names)
        all_categories.extend(categories)
        all_tools.extend(tools)
        total_tool_lists += tool_lists

    if len(set(all_names)) != len(all_names):
        raise ValueError("ToolSandbox base scenario names must be unique")

    source_files = {relative: _source_entry(root, relative) for relative in REQUIRED_SOURCE_FILES}
    # Base/no-distraction + three distraction levels + four schema-scramble variants.
    augmentation_count = 1 + 3 + 4
    payload: dict[str, Any] = {
        "kind": "localagent_toolsandbox_metadata_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision_url": f"{SOURCE_URL}/tree/{revision}",
        "revision": revision,
        "license": LICENSE,
        "source": {
            "checkout_root": root.name,
            "files": source_files,
            "metadata_only": True,
            "prompts_retained": False,
            "verifiers_read": False,
            "tools_executed": False,
            "user_simulator_executed": False,
            "external_api_called": False,
        },
        "coverage": {
            "base_scenarios": len(all_names),
            "unique_base_scenarios": len(set(all_names)),
            "scenario_families": family_rows,
            "category_tokens": _counter(all_categories),
            "tool_allow_list_literals": total_tool_lists,
            "tool_allow_list_entries": len(all_tools),
            "unique_tools": len(set(all_tools)),
            "unique_tool_names": sorted(set(all_tools)),
            "source_level_augmentation_multiplier": augmentation_count,
            "source_level_augmented_scenarios": len(all_names) * augmentation_count,
            "augmentation_policy": {
                "distraction_variants_per_base": ["none", "three", "ten", "all"],
                "schema_scramble_variants_per_base": 4,
                "schema_scramble_variants": [
                    "tool_description",
                    "argument_type",
                    "argument_description",
                    "tool_name",
                ],
                "note": (
                    "Counts are derived from the upstream named_scenarios source. Runtime tool "
                    "similarity ranking can change the concrete augmented allow-lists."
                ),
            },
        },
        "runtime_requirements": {
            "python_package": True,
            "polars": True,
            "dill": True,
            "user_simulator": True,
            "stateful_databases": ["setting", "contact", "messaging", "reminder"],
            "milestone_evaluator": True,
            "external_api_keys": "required for optional RapidAPI search tools only",
        },
        "claim_boundary": (
            "Public ToolSandbox source inventory only. Base scenario source is parsed without "
            "imports, task execution, user simulation, tool calls, verifiers, screenshots, "
            "external APIs, training rows, or a model score. Native evaluation still requires "
            "the pinned simulator and complete upstream result coverage."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite metadata receipt")
    payload = profile(args.root, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
