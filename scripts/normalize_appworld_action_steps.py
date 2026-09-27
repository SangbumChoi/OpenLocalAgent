#!/usr/bin/env python3
"""Export the first user-facing AppWorld API action as a canonical Conversation.

AppWorld ground truth is executable Python and often begins with supervisor/account bootstrap
calls and authentication.  A 10M WebGPU model cannot emit a complete multi-thousand-token
program in one turn, so this adapter creates a bounded first-action task: it records the first
non-bootstrap ``apis.<app>.<api>(...)`` request from each public train/dev solution as a
``run_python`` call.  The raw JSONL stays outside Git; the manifest stores only hashes and task
identities.  Native replay must inject fixture authentication and use the AppWorld verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.schema import Conversation, Message, Role, ToolCall

_BOOTSTRAP_APPS = {"admin", "api_docs", "supervisor"}
_BOOTSTRAP_APIS = {
    "login",
    "logout",
    "reset_password",
    "send_password_reset_code",
    "send_verification_code",
    "signup",
    "verify_account",
}
_OMIT_ARGUMENTS = {"_system_datetime", "access_token", "password", "raise_on_failure"}


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _text_hash(value: str) -> dict[str, int | str]:
    encoded = value.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _literal(value: Any) -> str:
    """Return a deterministic, safe Python literal for trace arguments."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_literal(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        items = [f"{_literal(key)}: {_literal(value[key])}" for key in sorted(value)]
        return "{" + ", ".join(items) + "}"
    raise TypeError(f"unsupported AppWorld argument type: {type(value).__name__}")


def _run_python_spec():
    for tool in STANDARD_TOOLS:
        if tool.name == "run_python":
            return tool
    raise RuntimeError("STANDARD_TOOLS is missing run_python")


def _first_action(trace: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any], str]:
    for request in trace:
        app = str(request.get("app", ""))
        api = str(request.get("api", ""))
        if not app or not api or app in _BOOTSTRAP_APPS or api in _BOOTSTRAP_APIS:
            continue
        arguments = {
            str(key): value
            for key, value in request.get("arguments", {}).items()
            if str(key) not in _OMIT_ARGUMENTS
        }
        code = f"apis.{app}.{api}({', '.join(f'{key}={_literal(arguments[key])}' for key in sorted(arguments))})"
        return app, api, arguments, code
    raise ValueError("ground-truth trace has no non-bootstrap AppWorld API action")


def _trace_ground_truth(world: Any) -> tuple[list[dict[str, Any]], str, bool]:
    ground_truth = world.task.ground_truth
    if ground_truth is None:
        raise ValueError(f"ground truth missing for public task {world.task_id}")
    captured: list[dict[str, Any]] = []
    original_request = world.requester.request

    def capture(*args: Any, **kwargs: Any):
        captured.append(
            {
                "app": kwargs.get("_app_name"),
                "api": kwargs.get("_api_name"),
                "arguments": {
                    str(key): value
                    for key, value in kwargs.items()
                    if key not in {"_app_name", "_api_name"}
                },
            }
        )
        return original_request(*args, **kwargs)

    world.requester.request = capture
    code = str(ground_truth.compiled_solution_code).rstrip() + "\nsolution(apis, requester)"
    execution = world.execute(code)
    tracker = world.evaluate()
    if not tracker.success:
        raise ValueError(f"public ground truth verifier failed for {world.task_id}")
    return captured, code, execution == "Execution successful."


def normalize(
    *, root: Path, split: str, purpose: str, output: Path, manifest: Path, max_tasks: int | None
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise ValueError("only AppWorld train/dev splits may be normalized; test splits are protected")
    if purpose not in {"train", "eval"}:
        raise ValueError("purpose must be train or eval")
    if (purpose, split) not in {("train", "train"), ("eval", "dev")}:
        raise ValueError("training export must use train; evaluation export must use disjoint dev")
    root = root.resolve()
    split_path = root / "data" / "datasets" / f"{split}.txt"
    version_path = root / "data" / "version.txt"
    if not split_path.is_file() or not version_path.is_file():
        raise FileNotFoundError("AppWorld data root must contain data/datasets and data/version.txt")

    try:
        from appworld import AppWorld, update_root
    except ImportError as error:
        raise RuntimeError("AppWorld must be installed in the optional evaluation environment") from error
    if Path(update_root(str(root))).resolve() != root:
        raise RuntimeError("AppWorld root did not resolve to the requested data root")
    try:
        appworld_version = importlib.metadata.version("appworld")
    except importlib.metadata.PackageNotFoundError:
        appworld_version = "unknown"

    task_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        task_ids = task_ids[:max_tasks]
    if not task_ids:
        raise ValueError(f"AppWorld {split} split is empty")

    tool = _run_python_spec()
    rows: list[str] = []
    task_sources: list[dict[str, Any]] = []
    for task_id in task_ids:
        spec_path = root / "data" / "tasks" / task_id / "specs.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"missing AppWorld task specs: {spec_path}")
        with AppWorld(
            task_id=task_id,
            experiment_name=f"localagent_action_step_{purpose}",
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=256,
        ) as world:
            instruction = str(world.task.instruction)
            trace, solution_code, execution_ok = _trace_ground_truth(world)
        if not execution_ok:
            raise ValueError(f"ground-truth execution did not complete cleanly for {task_id}")
        app, api, arguments, action_code = _first_action(trace)
        row = Conversation(
            messages=[
                Message(role=Role.user, content=instruction),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="run_python", arguments={"code": action_code})],
                ),
            ],
            tools=[tool],
            meta={
                "kind": "appworld_first_action_program",
                "parent_record_id": task_id,
                "source_dataset": "appworld",
                "source_split": split,
                "purpose": purpose,
                "data_version": version_path.read_text(encoding="utf-8").strip(),
                "instruction": _text_hash(instruction),
                "solution": _text_hash(solution_code),
                "first_action": {"app": app, "api": api, "arguments": _text_hash(_literal(arguments))},
            },
        )
        rows.append(row.to_json())
        task_sources.append(
            {
                "task_id": task_id,
                "spec": _sha256(spec_path),
                "instruction": _text_hash(instruction),
                "solution": _text_hash(solution_code),
                "first_action": {
                    "app": app,
                    "api": api,
                    "arguments": _text_hash(_literal(arguments)),
                    "code": _text_hash(action_code),
                },
            }
        )

    if output.exists() or output.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise FileExistsError("refusing to overwrite AppWorld action-step output or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    body = {
        "kind": "localagent_appworld_action_step_export",
        "schema_version": 1,
        "source": {
            "dataset": "appworld",
            "package_version": appworld_version,
            "data_version": version_path.read_text(encoding="utf-8").strip(),
            "root": str(root),
            "split": split,
            "split_file": _sha256(split_path),
            "purpose": purpose,
        },
        "output": _sha256(output),
        "rows": len(rows),
        "tasks": task_sources,
        "claim_boundary": (
            "Public AppWorld train/dev first non-bootstrap API actions normalized as run_python "
            "supervision. No protected test split or raw solution code is committed; this is an "
            "action-step adapter, not a complete AppWorld trajectory or leaderboard score."
        ),
    }
    body["manifest_self_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("APPWORLD_ROOT", ".")))
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--purpose", choices=("train", "eval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    result = normalize(
        root=args.root,
        split=args.split,
        purpose=args.purpose,
        output=args.output,
        manifest=args.manifest,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
