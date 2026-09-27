#!/usr/bin/env python3
"""Export bounded public AppWorld ground-truth API trajectories as Conversations.

This adapter keeps the public train/dev split boundary and removes bootstrap credentials.  Each
non-bootstrap API request becomes a ``run_python`` assistant turn followed by a redacted tool
observation.  It is deliberately a trajectory-learning projection: request arguments are taken
from the public ground-truth execution trace, while response bodies are summarized to avoid copying
task databases into the training artifact.  Native task success must still be measured separately.
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
_OMIT_ARGUMENTS = {
    "_system_datetime",
    "access_token",
    "password",
    "raise_on_failure",
    "username",
}
_SENSITIVE_KEYS = {
    "access_token", "api_key", "authorization", "canary", "cookie", "email", "password",
    "phone", "phone_number", "secret", "session", "token", "username",
}
_LOW_VALUE_RESPONSE_KEYS = {
    "birthday", "created_at", "updated_at", "url", "website",
}
_RELEVANT_RESPONSE_KEYS = {
    "album", "album_id", "album_title", "amount", "answer", "artist", "artists", "count",
    "contact_id", "description", "genre", "id", "last_name", "first_name", "message", "name",
    "page_index", "play_count", "playlist_id", "playlist_title", "rating", "relationships",
    "request_id", "response", "result", "sender", "receiver", "song_id", "status", "text",
    "text_message_id", "title", "total", "transaction_id", "type", "value", "follower_count",
}


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
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        suffix = "," if len(value) == 1 else ""
        return "(" + ", ".join(_literal(item) for item in value) + suffix + ")"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return "{" + ", ".join(f"{_literal(key)}: {_literal(value[key])}" for key in sorted(value)) + "}"
    raise TypeError(f"unsupported AppWorld argument type: {type(value).__name__}")


def _safe_value(value: Any, *, depth: int = 0, key: str = "") -> Any:
    """Keep useful schema/value hints while removing credentials and large task payloads."""

    normalized_key = key.lower()
    if (
        normalized_key in _SENSITIVE_KEYS
        or any(part in normalized_key for part in ("access_token", "password", "secret"))
        or normalized_key.endswith("_address")
        or normalized_key == "address"
        or normalized_key in _LOW_VALUE_RESPONSE_KEYS
    ):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:80] + ("…" if len(value) > 80 else "")
    if depth >= 3:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        items = sorted(
            (str(name), item)
            for name, item in value.items()
            if str(name).lower() in _RELEVANT_RESPONSE_KEYS
            or str(name).lower().endswith("_id")
        )
        kept = items[:12]
        result = {name: _safe_value(item, depth=depth + 1, key=name) for name, item in kept}
        if len(items) > len(kept):
            result["_truncated_fields"] = len(items) - len(kept)
        return result
    if isinstance(value, (list, tuple)):
        kept = list(value[:3])
        result = [_safe_value(item, depth=depth + 1) for item in kept]
        if len(value) > len(kept):
            result.append({"_truncated_items": len(value) - len(kept)})
        return result
    return str(value)[:80]


def _run_python_spec():
    for tool in STANDARD_TOOLS:
        if tool.name == "run_python":
            return tool
    raise RuntimeError("STANDARD_TOOLS is missing run_python")


def _trace_ground_truth(world: Any) -> tuple[list[dict[str, Any]], str]:
    ground_truth = world.task.ground_truth
    if ground_truth is None:
        raise ValueError(f"ground truth missing for {world.task_id}")
    captured: list[dict[str, Any]] = []
    original_request = world.requester.request

    def capture(*args: Any, **kwargs: Any):
        response = original_request(*args, **kwargs)
        captured.append(
            {
                "app": kwargs.get("_app_name"),
                "api": kwargs.get("_api_name"),
                "arguments": {
                    str(key): value
                    for key, value in kwargs.items()
                    if key not in {"_app_name", "_api_name"}
                },
                "response": response,
            }
        )
        return response

    world.requester.request = capture
    solution_code = str(ground_truth.compiled_solution_code).rstrip() + "\nsolution(apis, requester)"
    execution = world.execute(solution_code)
    if execution != "Execution successful.":
        raise ValueError(f"ground-truth execution failed for {world.task_id}: {execution!r}")
    if not world.evaluate().success:
        raise ValueError(f"ground-truth verifier failed for {world.task_id}")
    return captured, solution_code


def _actions(
    trace: list[dict[str, Any]], max_actions: int, *, include_completion: bool = False
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for request in trace:
        app = str(request.get("app", ""))
        api = str(request.get("api", ""))
        if not app or not api:
            continue
        is_completion = app == "supervisor" and api == "complete_task"
        if is_completion and include_completion:
            arguments = {
                str(key): value
                for key, value in request.get("arguments", {}).items()
                if str(key) not in _OMIT_ARGUMENTS
            }
            actions.append({"app": app, "api": api, "arguments": arguments, "response": request.get("response")})
            if len(actions) >= max_actions:
                break
            continue
        if app in _BOOTSTRAP_APPS or api in _BOOTSTRAP_APIS:
            continue
        arguments = {
            str(key): value
            for key, value in request.get("arguments", {}).items()
            if str(key) not in _OMIT_ARGUMENTS
        }
        actions.append(
            {
                "app": app,
                "api": api,
                "arguments": arguments,
                "response": request.get("response"),
            }
        )
        if len(actions) >= max_actions:
            break
    if not actions:
        raise ValueError("ground-truth trace has no non-bootstrap API action")
    return actions


def normalize(
    *, root: Path, split: str, purpose: str, output: Path, manifest: Path,
    max_tasks: int | None, max_actions: int, rich_observations: bool,
    selected_tasks: list[str] | None = None, include_completion: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise ValueError("only AppWorld train/dev splits may be normalized")
    if purpose not in {"train", "eval"} or (purpose, split) not in {("train", "train"), ("eval", "dev")}:
        raise ValueError("purpose/split must be train/train or eval/dev")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
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
    split_task_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if selected_tasks:
        unknown = sorted(set(selected_tasks) - set(split_task_ids))
        if unknown:
            raise ValueError(f"selected tasks are not in public {split} split: {unknown}")
        task_ids = list(dict.fromkeys(selected_tasks))
    else:
        task_ids = split_task_ids
    if max_tasks is not None and not selected_tasks:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        task_ids = task_ids[:max_tasks]
    tool = _run_python_spec()
    rows: list[str] = []
    task_sources: list[dict[str, Any]] = []
    for task_id in task_ids:
        with AppWorld(
            task_id=task_id,
            experiment_name=f"localagent_trajectory_{purpose}",
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=256,
        ) as world:
            instruction = str(world.task.instruction)
            trace, solution_code = _trace_ground_truth(world)
        actions = _actions(trace, max_actions, include_completion=include_completion)
        messages = [Message(role=Role.user, content=instruction)]
        for index, action in enumerate(actions):
            label = f"{action['app']}.{action['api']}"
            args = action["arguments"]
            code = f"apis.{label}({', '.join(f'{key}={_literal(args[key])}' for key in sorted(args))})"
            messages.append(
                Message(role=Role.assistant, tool_calls=[ToolCall(name="run_python", arguments={"code": code})])
            )
            observation: dict[str, Any] = {"status": "ok", "step": index, "api": label}
            if rich_observations:
                observation["response"] = _safe_value(action.get("response"))
            messages.append(
                Message(
                    role=Role.tool,
                    tool_response=json.dumps(observation, separators=(",", ":"), sort_keys=True),
                )
            )
        row = Conversation(
            messages=messages,
            tools=[tool],
            meta={
                "kind": "appworld_ground_truth_api_trajectory",
                "parent_record_id": task_id,
                "source_dataset": "appworld",
                "source_split": split,
                "purpose": purpose,
                "data_version": version_path.read_text(encoding="utf-8").strip(),
                "trajectory_steps": len(actions),
                "trajectory_truncated": len(actions) < sum(
                    int(
                        (
                            str(item.get("app", "")) == "supervisor"
                            and str(item.get("api", "")) == "complete_task"
                            and include_completion
                        )
                        or (
                            str(item.get("app", "")) not in _BOOTSTRAP_APPS
                            and str(item.get("api", "")) not in _BOOTSTRAP_APIS
                        )
                    )
                    for item in trace
                ),
                "instruction": _text_hash(instruction),
                "solution": _text_hash(solution_code),
                "action_labels": [f"{item['app']}.{item['api']}" for item in actions],
            },
        )
        rows.append(row.to_json())
        task_sources.append(
            {
                "task_id": task_id,
                "spec": _sha256(root / "data" / "tasks" / task_id / "specs.json"),
                "instruction": _text_hash(instruction),
                "steps": len(actions),
                "truncated": bool(row.meta["trajectory_truncated"]),
                "labels": row.meta["action_labels"],
            }
        )
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite trajectory output or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    body = {
        "kind": "localagent_appworld_trajectory_export",
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
        "configuration": {
            "max_tasks": max_tasks,
            "max_actions": max_actions,
            "bootstrap_credentials_removed": True,
            "tool_observations": (
                "bounded safe response summaries with credentials/canaries removed"
                if rich_observations else "redacted status/api/step summaries"
            ),
            "rich_observations": rich_observations,
            "include_completion": include_completion,
        },
        "output": _sha256(output),
        "rows": len(rows),
        "tasks": task_sources,
        "claim_boundary": (
            "Public AppWorld train/dev ground-truth API trajectories with bootstrap credentials removed. "
            "When rich observations are enabled, responses are bounded safe summaries with sensitive "
            "keys/canaries redacted; this is trajectory-learning supervision, not a native AppWorld "
            "score or a claim that summaries reproduce the full stateful environment."
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
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--task", dest="tasks", action="append", help="select an exact public split task (repeatable)")
    parser.add_argument("--include-completion", action="store_true", help="retain supervisor.complete_task as the final action")
    parser.add_argument(
        "--rich-observations",
        action="store_true",
        help="include bounded response values/keys with credentials and canaries removed",
    )
    args = parser.parse_args()
    result = normalize(
        root=args.root,
        split=args.split,
        purpose=args.purpose,
        output=args.output,
        manifest=args.manifest,
        max_tasks=args.max_tasks,
        max_actions=args.max_actions,
        rich_observations=args.rich_observations,
        selected_tasks=args.tasks,
        include_completion=args.include_completion,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
