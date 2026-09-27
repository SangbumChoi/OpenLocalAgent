#!/usr/bin/env python3
"""Run bounded MCPMark Verified Playwright tasks with a LocalAgent checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SOURCE_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"
PLAYWRIGHT_PACKAGE = "@playwright/mcp@0.0.68"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def discover_tasks(root: Path, requested: list[str]) -> list[tuple[str, str, Path, Path]]:
    found: dict[str, tuple[str, str, Path, Path]] = {}
    for verifier in sorted((root / "tasks" / "playwright" / "standard").glob("*/*/verify.py")):
        task_dir = verifier.parent
        description = task_dir / "description.md"
        if description.exists():
            category = task_dir.parent.name
            task_id = task_dir.name
            found[f"{category}/{task_id}"] = (category, task_id, description, verifier)
    if requested:
        missing = [name for name in requested if name not in found]
        if missing:
            raise ValueError(f"requested tasks not found: {missing}")
        return [found[name] for name in requested]
    return list(found.values())


def _load_stdio(root: Path):
    module_path = root / "src" / "agents" / "mcp" / "stdio_server.py"
    spec = importlib.util.spec_from_file_location("mcpmark_playwright_stdio", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MCPMark wrapper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MCPStdioServer


async def _run_one(
    task: tuple[str, str, Path, Path],
    *,
    root: Path,
    checkpoint: Path,
    executable: Path,
    max_turns: int,
) -> dict[str, Any]:
    from localagent.agent.constrained import hybrid_decode
    from localagent.agent.parser import extract_tool_calls
    from localagent.agent.retriever import ToolRetriever
    from localagent.agent.runtime import Agent
    from localagent.agent.tools import ToolRegistry
    from localagent.data.schema import ToolSpec

    category, task_id, description_path, verifier_path = task
    MCPStdioServer = _load_stdio(root)
    args = [
        "-y",
        PLAYWRIGHT_PACKAGE,
        "--headless",
        "--isolated",
        "--no-sandbox",
        "--browser",
        "chromium",
        "--executable-path",
        str(executable),
        "--viewport-size",
        "1280,720",
    ]
    turns: list[dict[str, Any]] = []
    server_error: str | None = None
    verifier_exit_code: int | None = None
    verifier_stdout = ""
    verifier_stderr = ""
    with tempfile.TemporaryDirectory(prefix=f"mcpmark-playwright-{category}-{task_id}-", dir="/private/tmp") as td:
        messages_path = Path(td) / "messages.json"
        try:
            async with MCPStdioServer(command="npx", args=args, timeout=120) as server:
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
                agent = Agent.from_checkpoint(checkpoint, registry, retriever=ToolRetriever(specs))
                prompt = (
                    f"USER: {description_path.read_text()}\n"
                    "Use the available Playwright browser tools. Perform the task in the browser, "
                    "then provide the requested final answer and stop."
                )
                messages: list[dict[str, Any]] = []
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
                    messages.append(
                        {
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                            "content": [{"type": "text", "text": output}],
                        }
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
                            "result_preview": str(result.get("content", ""))[:300],
                        }
                    )
                    turns.append(row)
                    messages.append({"role": "tool", "name": call.name, "content": result})
                    prompt += f"\nASSISTANT: {output}\nTOOL_RESULT: {json.dumps(result, sort_keys=True)}"
                messages_path.write_text(json.dumps(messages, sort_keys=True))
        except Exception as exc:
            server_error = f"{type(exc).__name__}: {exc}"
            messages_path.write_text("[]")

        env = os.environ.copy()
        env["MCP_MESSAGES"] = str(messages_path)
        proc = subprocess.run(
            [sys.executable, str(verifier_path)], capture_output=True, text=True, env=env, check=False
        )
        verifier_exit_code = proc.returncode
        verifier_stdout = proc.stdout
        verifier_stderr = proc.stderr
    return {
        "category": category,
        "task_id": task_id,
        "task": f"playwright/standard/{category}/{task_id}",
        "turns": turns,
        "server_error": server_error,
        "verifier_exit_code": verifier_exit_code,
        "verifier_stdout_sha256": hashlib.sha256(verifier_stdout.encode()).hexdigest(),
        "verifier_stderr_sha256": hashlib.sha256(verifier_stderr.encode()).hexdigest(),
        "verifier_stdout_tail": verifier_stdout[-500:],
        "verifier_stderr_tail": verifier_stderr[-500:],
    }


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    tasks = discover_tasks(args.mcpmark_root, args.task)
    if args.limit:
        tasks = tasks[: args.limit]
    results = []
    for task in tasks:
        row = await _run_one(
            task,
            root=args.mcpmark_root,
            checkpoint=args.checkpoint,
            executable=args.executable,
            max_turns=args.max_turns,
        )
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    passes = sum(r["verifier_exit_code"] == 0 and not r["server_error"] for r in results)
    return {
        "kind": "localagent_mcpmark_native_playwright_standard_current_checkpoint",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "source_revision": SOURCE_REVISION,
        "official_split_verified": False,
        "task_count": len(results),
        "success_rate": passes / len(results) if results else 0.0,
        "checkpoint": _identity(args.checkpoint),
        "dataset": {
            "name": "MCPMark Verified",
            "url": "https://github.com/eval-sys/mcpmark",
            "revision": SOURCE_REVISION,
            "suite": "standard",
            "service": "playwright",
            "scope": "bounded_public_browser_service_diagnostic",
        },
        "environment": {
            "mcp_server_executed": True,
            "server_package": PLAYWRIGHT_PACKAGE,
            "server_command": ["npx", *args.playwright_args],
            "browser_executable": _identity(args.executable),
            "browser_version": "Chromium 125.0.6422.26",
            "official_split_verified": False,
            "user_simulator_executed": False,
            "external_api_called": False,
        },
        "results": results,
        "summary": {
            "tasks": len(results),
            "verifier_passes": passes,
            "verifier_failures": len(results) - passes,
            "runtime_errors": sum(bool(r["server_error"]) for r in results),
            "browser_tool_errors": sum(
                1 for r in results for turn in r["turns"] if turn.get("result_is_error")
            ),
        },
        "claim_boundary": (
            "This bounded diagnostic exercised the real pinned Playwright MCP server and a local "
            "Chromium executable on selected public MCPMark Verified tasks. It is not an official "
            "MCPMark score: the full split, browser-version parity, visual answer quality, user "
            "simulation, WebArena services, and external accounts were not executed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcpmark-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()
    args.playwright_args = [
        "-y", PLAYWRIGHT_PACKAGE, "--headless", "--isolated", "--no-sandbox", "--browser", "chromium",
        "--executable-path", str(args.executable), "--viewport-size", "1280,720",
    ]
    payload = asyncio.run(_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
