#!/usr/bin/env python3
"""Run the pinned MCPMark filesystem/standard service subset with a checkpoint.

The runner deliberately keeps the MCP server and verifier inside a temporary workspace.  It
records hashes and verifier exits rather than exposing tool output, so the resulting receipt is
safe to publish and reproducible without external accounts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SOURCE_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"
SERVER_PACKAGE = "@modelcontextprotocol/server-filesystem"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def discover_tasks(root: Path, suite: str = "standard") -> list[tuple[str, str, Path, Path]]:
    out = []
    for verifier in sorted((root / "tasks" / "filesystem" / suite).glob("*/*/verify.py")):
        task_dir = verifier.parent
        description = task_dir / "description.md"
        if description.exists():
            out.append((task_dir.parent.name, task_dir.name, description, verifier))
    return out


def _state_source(category: str, primary: Path, fallback: Path | None) -> Path:
    for base in (primary, fallback):
        if base is not None and (base / category).is_dir():
            return base / category
    raise FileNotFoundError(f"no state archive extracted for category {category}")


async def _run_one(
    task: tuple[str, str, Path, Path],
    *,
    checkpoint: Path,
    mcp_root: Path,
    primary_state: Path,
    fallback_state: Path | None,
    max_turns: int,
) -> dict[str, Any]:
    from localagent.agent.constrained import hybrid_decode
    from localagent.agent.parser import extract_tool_calls
    from localagent.agent.retriever import ToolRetriever
    from localagent.agent.runtime import Agent
    from localagent.agent.tools import ToolRegistry
    from localagent.data.schema import ToolSpec

    category, task_id, description_path, verifier_path = task
    with tempfile.TemporaryDirectory(prefix=f"mcpmark-standard-{category}-{task_id}-", dir="/private/tmp") as td:
        work = Path(td) / category
        shutil.copytree(_state_source(category, primary_state, fallback_state), work)
        server_error = None
        turns: list[dict[str, Any]] = []
        try:
            module_path = mcp_root / "src" / "agents" / "mcp" / "stdio_server.py"
            spec = importlib.util.spec_from_file_location("mcpmark_stdio_server", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load MCPMark stdio wrapper: {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MCPStdioServer = module.MCPStdioServer

            async with MCPStdioServer(
                command="npx",
                args=["--yes", SERVER_PACKAGE, str(work)],
                timeout=120,
            ) as server:
                raw_tools = await server.list_tools()
                specs = [
                    ToolSpec(
                        name=str(tool["name"]),
                        description=str(tool.get("description", "")),
                        parameters=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    )
                    for tool in raw_tools
                ]
                registry = ToolRegistry()
                for spec in specs:
                    registry.register(spec, lambda **_: None)
                agent = Agent.from_checkpoint(
                    checkpoint, registry, retriever=ToolRetriever(specs)
                )
                prompt = (
                    f"USER: {description_path.read_text()}\n"
                    f"Workspace root: {work}\n"
                    "Use the available MCP filesystem tools. Complete the task, then stop."
                )
                for turn in range(1, max_turns + 1):
                    output = hybrid_decode(
                        agent.model,
                        agent.tokenizer,
                        prompt,
                        specs,
                        "cpu",
                        retriever=ToolRetriever(specs),
                        route_head=agent.route_head,
                        selector=agent.selector,
                        ptr_head=agent.ptr_head,
                        top_m=1,
                    )
                    calls = extract_tool_calls(output)
                    call = calls[0] if calls else None
                    row: dict[str, Any] = {
                        "turn": turn,
                        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                        "tool": call.name if call else None,
                        "arguments_sha256": _sha(call.arguments) if call else None,
                    }
                    if call is None:
                        turns.append(row)
                        break
                    result = await server.call_tool(call.name, call.arguments)
                    row.update(
                        {
                            "result_is_error": bool(result.get("isError", False)),
                            "result_sha256": _sha(result),
                        }
                    )
                    turns.append(row)
                    prompt += f"\nASSISTANT: {output}\nTOOL_RESULT: {json.dumps(result, sort_keys=True)}"
        except Exception as exc:  # keep one broken task from invalidating the receipt
            server_error = f"{type(exc).__name__}: {exc}"

        env = os.environ.copy()
        env["FILESYSTEM_TEST_DIR"] = str(work)
        proc = subprocess.run(
            [sys.executable, str(verifier_path)], capture_output=True, text=True, env=env, check=False
        )
        return {
            "category": category,
            "task_id": task_id,
            "task": f"filesystem/standard/{category}/{task_id}",
            "turns": turns,
            "server_error": server_error,
            "verifier_exit_code": proc.returncode,
            "verifier_stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
            "verifier_stderr_sha256": hashlib.sha256(proc.stderr.encode()).hexdigest(),
            "verifier_stdout_tail": proc.stdout[-500:],
            "verifier_stderr_tail": proc.stderr[-500:],
        }


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    tasks = discover_tasks(args.mcpmark_root)
    if args.limit:
        tasks = tasks[: args.limit]
    results = []
    for task in tasks:
        row = await _run_one(
            task,
            checkpoint=args.checkpoint,
            mcp_root=args.mcpmark_root,
            primary_state=args.primary_state_root,
            fallback_state=args.fallback_state_root,
            max_turns=args.max_turns,
        )
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    passes = sum(r["verifier_exit_code"] == 0 and not r["server_error"] for r in results)
    ck_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    return {
        "kind": "localagent_mcpmark_native_filesystem_standard_current_checkpoint",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "source_revision": SOURCE_REVISION,
        "official_split_verified": True,
        "task_count": len(results),
        "success_rate": passes / len(results) if results else 0.0,
        "checkpoint": {"path": str(args.checkpoint), "sha256": ck_sha},
        "dataset": {
            "name": "MCPMark Verified",
            "url": "https://github.com/eval-sys/mcpmark",
            "revision": SOURCE_REVISION,
            "suite": "standard",
            "service": "filesystem",
            "scope": "official_standard_service_subset",
        },
        "environment": {
            "mcp_server_executed": True,
            "server_package": SERVER_PACKAGE,
            "server_command": ["npx", "--yes", SERVER_PACKAGE],
            "user_simulator_executed": False,
            "external_api_called": False,
        },
        "results": results,
        "summary": {
            "tasks": len(results),
            "verifier_passes": passes,
            "verifier_failures": len(results) - passes,
            "runtime_errors": sum(bool(r["server_error"]) for r in results),
        },
        "claim_boundary": (
            "All pinned filesystem/standard tasks in this official service subset ran against the "
            "version-pinned MCPMark task/verifier tree and an isolated stdio MCP filesystem server. "
            "This is not a complete cross-service MCPMark score; Notion, GitHub, Postgres, "
            "Playwright, user simulation, and external accounts were not executed."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mcpmark-root", type=Path, required=True)
    p.add_argument("--primary-state-root", type=Path, required=True)
    p.add_argument("--fallback-state-root", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    payload = asyncio.run(_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
