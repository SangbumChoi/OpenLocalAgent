#!/usr/bin/env python3
"""Profile the public MobileWorld source and host prerequisites without running a task.

The benchmark needs a privileged Docker-in-Docker Android emulator, KVM acceleration, and
optional model/MCP/user-agent credentials.  This profiler only parses the task definitions and
records source hashes plus local prerequisite availability; it never launches a container, ADB
session, MCP service, model, or user-agent simulator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = "Tongyi-MAI/MobileWorld"
SOURCE_URL = "https://github.com/Tongyi-MAI/MobileWorld"
EXPECTED_TASKS = 201


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _is_base_task(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseTask":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseTask":
            return True
    return False


def _task_inventory(root: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    task_root = root / "src/mobile_world/tasks/definitions"
    if not task_root.is_dir():
        raise ValueError(f"MobileWorld task definitions are missing: {task_root}")
    domain_counts: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    for path in sorted(task_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = sum(isinstance(node, ast.ClassDef) and _is_base_task(node) for node in ast.walk(tree))
        if count:
            domain = path.relative_to(task_root).parts[0]
            domain_counts[domain] += count
        files.append({**_identity(root, path), "task_classes": count})
    return dict(sorted(domain_counts.items())), files


def profile(root: Path, *, revision: str) -> dict[str, Any]:
    domains, task_files = _task_inventory(root)
    task_count = sum(domains.values())
    if task_count != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} MobileWorld tasks, found {task_count}")

    required_files = [
        root / "README.md",
        root / "LICENSE",
        root / "pyproject.toml",
        root / ".env.example",
        root / "src/mobile_world/tasks/registry.py",
    ]
    source_files = [_identity(root, path) for path in required_files if path.is_file()]
    task_digest = hashlib.sha256(
        "\n".join(
            f"{row['path']}:{row['sha256']}:{row['task_classes']}" for row in task_files
        ).encode("utf-8")
    ).hexdigest()
    host = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "docker_command": shutil.which("docker") is not None,
        "adb_command": shutil.which("adb") is not None,
        "qemu_command": shutil.which("qemu-system-x86_64") is not None,
        "kvm_device": Path("/dev/kvm").is_char_device(),
    }
    payload: dict[str, Any] = {
        "kind": "localagent_mobileworld_source_runtime_audit",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": revision,
        "license": "Apache-2.0 (source repository; benchmark assets/runtime terms remain upstream-controlled)",
        "source": {
            "files": source_files,
            "task_definition_files": task_files,
            "task_definitions_sha256": task_digest,
        },
        "benchmark_contract": {
            "tasks": task_count,
            "apps": 20,
            "domains": domains,
            "features": [
                "long_horizon_cross_app_workflows",
                "agent_user_interaction",
                "mcp_augmented_tasks",
                "deterministic_backend_and_local_storage_verification",
            ],
            "observation": ["screenshot", "accessibility_tree", "backend_state"],
            "actions": ["tap", "swipe", "type", "keyevent", "wait", "mcp_call"],
        },
        "runtime_requirements": {
            "host": "Linux or WSL2 with KVM; macOS support is listed as in progress upstream",
            "docker_privileged": True,
            "rooted_android_avd": True,
            "model_api_credentials": True,
            "user_agent_credentials_for_interactive_tasks": True,
            "mcp_credentials_for_mcp_tasks": True,
        },
        "host_preflight": host,
        "execution": {
            "official_runner_executed": False,
            "native_environment_executed": False,
            "task_rows_copied_to_training": 0,
            "score": None,
        },
        "claim_boundary": (
            "Pinned MobileWorld source/task-definition audit only. The profiler parses the public "
            "task classes and checks local command availability; it does not launch the privileged "
            "Docker/KVM Android environment, use ADB, call MCP services, send email, or execute a "
            "model. No native score or training-data admission is claimed."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")
    payload = profile(args.root.resolve(), revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
